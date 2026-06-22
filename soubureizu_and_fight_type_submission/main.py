%%writefile main.py
import os
import sys
from collections import defaultdict

if not os.path.exists("cg"):
    sys.path.insert(0, "/kaggle_simulations/agent")

from cg.api import (
    AreaType, CardType, EnergyType, Observation, SelectContext, OptionType,
    Card, Pokemon, all_attack, all_card_data, to_observation_class,
)

"""
Soubureizu ex + Fighting Type Deck
Early game: Solrock attacks with コスモビーム (70, no weakness/resistance) while Lunatone uses
ルナサイクル to trash fighting energy and draw cards.
Late game: When enough fire energy is in discard, switch to Soubureizu_ex's しんえんほむら
for heavy damage.
"""

file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path
with open(file_path, "r") as f:
    csv = f.read().split("\n")
my_deck = [int(csv[i]) for i in range(60)]

all_card   = all_card_data()
card_table = {c.cardId: c for c in all_card}

# ─── Card IDs ───────────────────────────────────────────────
Charcadet       = 319
Soubureizu_ex   = 320
Solrock         = 676
Lunatone        = 675
Mogurew         = 81
Kitchigisu_ex   = 140

Fire_Energy     = 2
Fighting_Energy = 6
Mist_Energy     = 11

Hyperball       = 1121
Poke_Pad        = 1152
Fight_Gong      = 1142
Night_Stretcher = 1097
Perfect_Mixer   = 1128

Boss_Orders     = 1182
Lillie          = 1227
Zeigh           = 1192
Explorer        = 1185

# ケーシィ・ユンゲラー・フーディンのカードID（ミストエネルギー最優先発動条件）
ABRA_ALAKAZAM_LINE = {109, 741, 742, 245, 743}

# Precompute attack IDs
_atk_id       = {a.name: a.attackId for a in all_attack()}
ATK_SINHOMURA = _atk_id.get("しんえんほむら", -1)
ATK_COSMOBEAM = _atk_id.get("コスモビーム",   -1)

# しんえんほむらへの切り替えしきい値
SWITCH_FIRE_THRESHOLD = 3  # trash_fire >= 3 → Soubureizu_ex preferred (90 damage > 70)


class AttackPlan:
    attacker     = -1   # index in my_cards (0=active, 1+=bench)
    target       = -1   # index in op_cards
    attack_index = -1   # 0=コスモビーム, 1=しんえんほむら
    remain_hp    = -1
    energy       = False  # True if energy attachment still needed


plan             = AttackPlan()
pre_turn         = 0
op_attacker_ids: set = set()


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:    return obs.select.deck[index]
        case AreaType.HAND:    return ps.hand[index]
        case AreaType.DISCARD: return ps.discard[index]
        case AreaType.ACTIVE:  return ps.active[index]
        case AreaType.BENCH:   return ps.bench[index]
        case AreaType.PRIZE:   return ps.prize[index]
        case AreaType.STADIUM: return obs.current.stadium[index]
        case AreaType.LOOKING: return obs.current.looking[index]
        case _:                return None


def prize_count(pokemon: Pokemon) -> int:
    data  = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == 12:
            count -= 1
    return max(0, count)


def sinhomura_damage(trash_fire: int, target: Pokemon) -> int:
    damage = 30 + trash_fire * 20
    data   = card_table[target.id]
    if data.weakness == EnergyType.FIRE:
        damage *= 2
    return damage


def cosmobeam_damage() -> int:
    return 70  # fixed, no weakness / resistance calculation


def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    state    = obs.current
    select   = obs.select
    context  = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    global plan, pre_turn, op_attacker_ids
    if pre_turn != state.turn:
        pre_turn = state.turn
        plan     = AttackPlan()
        if op_state.active and op_state.active[0] is not None and len(op_state.active[0].energies) >= 1:
            op_attacker_ids.add(op_state.active[0].id)

    field_counts   = defaultdict(int)
    hand_counts    = defaultdict(int)
    discard_counts = defaultdict(int)

    for card in my_state.active + my_state.bench:
        if card is not None:
            field_counts[card.id] += 1
    for card in my_state.hand:
        hand_counts[card.id] += 1
    for card in my_state.discard:
        discard_counts[card.id] += 1

    trash_fire = discard_counts[Fire_Energy]

    # 優先アタッカー判断 (しんえんほむら switch threshold)
    preferred_attacker = Soubureizu_ex if trash_fire >= SWITCH_FIRE_THRESHOLD else Solrock

    # ドロー系サポートを守る条件
    bench_not_setup = (field_counts[Solrock] + field_counts[Lunatone] +
                       field_counts[Charcadet] + field_counts[Soubureizu_ex]) < 3
    no_energy_in_hand = (hand_counts[Fire_Energy] == 0 and hand_counts[Fighting_Energy] == 0)
    need_draw = bench_not_setup or no_energy_in_hand

    my_active = my_state.active[0] if my_state.active else None
    op_active = op_state.active[0]  if op_state.active else None
    my_cards  = ([my_active] if my_active is not None else []) + [p for p in my_state.bench if p is not None]
    op_cards  = ([op_active] if op_active is not None else []) + [p for p in op_state.bench  if p is not None]

    # 相手の場・ベンチにケーシィ・ユンゲラー・フーディンがいるか
    op_has_abra_line = any(p is not None and p.id in ABRA_ALAKAZAM_LINE for p in op_cards)

    # ─── Attack plan ──────────────────────────────────────────
    if context == SelectContext.MAIN and state.turn >= 2:
        can_switch    = False
        can_op_switch = False
        for o in select.option:
            if o.type == OptionType.PLAY:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                if card.id == Boss_Orders:
                    can_op_switch = True
            elif o.type == OptionType.RETREAT:
                can_switch = True

        best_score = -1

        for i, my_pokemon in enumerate(my_cards):
            if i != 0 and not can_switch:
                continue

            # ─── Solrock / コスモビーム ───
            if my_pokemon.id == Solrock:
                lunatone_on_bench = any(
                    p is not None and p.id == Lunatone for p in my_state.bench
                )
                if not lunatone_on_bench:
                    continue  # ルナトーンがベンチにいないとコスモビーム失敗

                fighting_ec = sum(1 for e in my_pokemon.energies if e == EnergyType.FIGHTING)
                more_energy = False
                if fighting_ec < 1:
                    if hand_counts[Fighting_Energy] >= 1 and not state.energyAttached:
                        fighting_ec += 1
                        more_energy  = True
                    else:
                        continue

                for j, op_pokemon in enumerate(op_cards):
                    if j != 0 and not can_op_switch:
                        break
                    damage = cosmobeam_damage()
                    score  = prize_count(op_pokemon) * 1000
                    if len(op_state.prize) <= prize_count(op_pokemon) and op_pokemon.hp <= damage:
                        score = 50000
                    elif op_pokemon.hp <= damage:
                        score += 2000
                    else:
                        score += int(1000 * damage / op_pokemon.hp)
                    score += 300 if j == 0 else 0
                    score += 200 if i == 0 else 0

                    if best_score < score:
                        best_score        = score
                        plan.attacker     = i
                        plan.target       = j
                        plan.attack_index = 0
                        plan.remain_hp    = op_pokemon.hp - damage
                        plan.energy       = more_energy

            # ─── Soubureizu_ex / しんえんほむら ───
            elif my_pokemon.id == Soubureizu_ex:
                fire_ec     = sum(1 for e in my_pokemon.energies if e == EnergyType.FIRE)
                more_energy = False
                if fire_ec < 1:
                    can_get_fire = (
                        hand_counts[Fire_Energy] >= 1 or
                        (hand_counts[Night_Stretcher] >= 1 and discard_counts[Fire_Energy] >= 1)
                    )
                    if can_get_fire and not state.energyAttached:
                        fire_ec     += 1
                        more_energy  = True
                    else:
                        continue

                for j, op_pokemon in enumerate(op_cards):
                    if j != 0 and not can_op_switch:
                        break
                    damage = sinhomura_damage(trash_fire, op_pokemon)
                    score  = prize_count(op_pokemon) * 1000
                    if len(op_state.prize) <= prize_count(op_pokemon) and op_pokemon.hp <= damage:
                        score = 50000
                    elif op_pokemon.hp <= damage:
                        score += 2000
                    else:
                        score += int(1000 * damage / op_pokemon.hp)
                    score += 300 if j == 0 else 0
                    score += 200 if i == 0 else 0

                    if best_score < score:
                        best_score        = score
                        plan.attacker     = i
                        plan.target       = j
                        plan.attack_index = 1
                        plan.remain_hp    = op_pokemon.hp - damage
                        plan.energy       = more_energy

        # ボスの指令: バトル場をOHKOできない場合、非exアタッカー（エネ持ち）を狙う
        if (can_op_switch and plan.attacker >= 0 and plan.target == 0 and plan.remain_hp > 0):
            my_att = my_cards[plan.attacker] if plan.attacker < len(my_cards) else None
            can_still_attack = False
            if my_att and my_att.id == Solrock:
                fc = sum(1 for e in my_att.energies if e == EnergyType.FIGHTING)
                can_still_attack = fc >= 1 or (hand_counts[Fighting_Energy] >= 1 and not state.energyAttached)
            elif my_att and my_att.id == Soubureizu_ex:
                fc = sum(1 for e in my_att.energies if e == EnergyType.FIRE)
                can_still_attack = fc >= 1 or (
                    not state.energyAttached and (
                        hand_counts[Fire_Energy] >= 1 or
                        (hand_counts[Night_Stretcher] >= 1 and discard_counts[Fire_Energy] >= 1)
                    )
                )

            if can_still_attack:
                for j, op_pokemon in enumerate(op_cards):
                    if j == 0:
                        continue
                    if op_pokemon.id not in op_attacker_ids:
                        continue
                    if len(op_pokemon.energies) == 0:
                        continue
                    data = card_table[op_pokemon.id]
                    if data.ex:
                        continue
                    damage = sinhomura_damage(trash_fire, op_pokemon)
                    score  = int(1000 * damage / op_pokemon.hp) + len(op_pokemon.energies) * 100 + 600
                    if op_pokemon.hp <= damage:
                        score += 2000
                    if score > best_score:
                        best_score        = score
                        plan.attacker     = 0
                        plan.target       = j
                        plan.attack_index = plan.attack_index
                        plan.remain_hp    = op_pokemon.hp - damage
                        plan.energy       = (my_att.id == Solrock and
                                             sum(1 for e in my_att.energies if e == EnergyType.FIGHTING) < 1)

    # ─── Energy attachment scoring ────────────────────────────
    def energy_score(energy_id: int | None, pokemon: Pokemon, active: bool) -> int:
        count = len(pokemon.energies)

        # ソウブレイズex
        if pokemon.id == Soubureizu_ex:
            fire_count = sum(1 for e in pokemon.energies if e == EnergyType.FIRE)
            mist_count = sum(1 for c in pokemon.energyCards if c.id == Mist_Energy)

            # ミストエネルギー: 相手にケーシィラインがいればソウブレイズexに最優先で付ける
            if energy_id == Mist_Energy and mist_count == 0 and op_has_abra_line:
                # 炎エネがすでにある → 攻撃可能なのでミストを最優先（進化10000+より高く）
                # 炎エネがない → 炎エネ付与(8010)を先にするためミストはやや低め
                return 10500 if fire_count >= 1 else 7900

            if fire_count >= 1:
                return 50  # 炎エネあり、追加不要
            if energy_id == Fire_Energy:
                return 8010 + (10 if active else 0)
            return 200

        # ソルロック: 闘エネ1枚（コスモビーム用）
        if pokemon.id == Solrock:
            fighting_count = sum(1 for e in pokemon.energies if e == EnergyType.FIGHTING)
            if fighting_count >= 1:
                return 100
            lunatone_on_bench = any(p is not None and p.id == Lunatone for p in my_state.bench)
            if energy_id == Fighting_Energy:
                return 8000 + (10 if active else 0) if lunatone_on_bench else 4000
            # ミストエネルギーはコスモビームのコスト（闘×1）を満たせない
            return 500

        # バトル場の非アタッカー: 優先アタッカーがベンチにいる場合に逃げコスト払うため
        non_attacker_ids = {Charcadet, Lunatone, Mogurew, Kitchigisu_ex}
        if pokemon.id in non_attacker_ids and active and count == 0:
            preferred_on_bench = any(
                p is not None and p.id == preferred_attacker for p in my_state.bench
            )
            if preferred_on_bench:
                if pokemon.id == Charcadet:
                    can_evolve = hand_counts[Soubureizu_ex] > 0
                    return 8800 if not can_evolve else 3000
                return 8800  # Lunatone/Mogurew/Kitchigisu_ex も逃げるため

        if pokemon.id == Lunatone:
            return 400  # ルナサイクルで手札から捨てる→場につけるより優先度低
        if pokemon.id == Mogurew:
            return 400
        if pokemon.id == Kitchigisu_ex:
            return 300
        return 500

    # ─── Score every option ────────────────────────────────────
    scores = []
    for o in select.option:
        score = 0

        if o.type == OptionType.NUMBER:
            score = o.number

        elif o.type == OptionType.YES:
            score = 1

        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is not None:
                energy_count = len(card.energies) if isinstance(card, Pokemon) else 0

                if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
                    if o.playerIndex == my_index:
                        # 優先アタッカーをバトル場に出す
                        if card.id == preferred_attacker:
                            score = 120 + energy_count
                        elif card.id in (Solrock, Soubureizu_ex):
                            score = 110 + energy_count
                        elif card.id == Charcadet:
                            score = 50 + energy_count
                        else:
                            score = 10 + energy_count
                        if plan.attacker > 0 and o.area == AreaType.BENCH and o.index == plan.attacker - 1:
                            score += 200
                    else:
                        if plan.target > 0 and o.index == plan.target - 1:
                            score += 200

                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    if card.id == Solrock:
                        score = 10  # メインアタッカー
                    elif card.id == Charcadet:
                        score = 8   # ソウブレイズexへ進化
                    elif card.id == Lunatone:
                        score = 6
                    elif card.id == Mogurew:
                        score = 5
                    else:
                        score = 3

                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    if card.id == Lunatone:
                        score = 10  # ルナサイクル使用に必須
                    elif card.id == Solrock:
                        score = 9
                    elif card.id == Charcadet:
                        score = 7
                    elif card.id == Mogurew:
                        score = 6
                    else:
                        score = 4

                elif context == SelectContext.TO_HAND:
                    score = 200 - hand_counts[card.id] * 50
                    if card.id == Solrock:
                        score += 100 if field_counts[Solrock] < 1 else 0
                    elif card.id == Lunatone:
                        score += 100 if field_counts[Lunatone] < 1 else 0
                    elif card.id == Charcadet:
                        score += 50 if (field_counts[Charcadet] + field_counts[Soubureizu_ex]) < 2 else -50
                    elif card.id == Soubureizu_ex:
                        score += 80 if field_counts[Soubureizu_ex] < 1 else -100
                    elif card.id == Fire_Energy:
                        is_tanker = select.effect is not None and select.effect.id == Night_Stretcher
                        if is_tanker and field_counts[Soubureizu_ex] >= 1 and all(
                            sum(1 for e in p.energies if e == EnergyType.FIRE) == 0
                            for p in my_cards if p.id == Soubureizu_ex
                        ):
                            score += 300  # 夜のタンカで攻撃用炎エネを優先取得 (200+300=500)
                        else:
                            score -= 100  # 通常時はトラッシュに残す
                    elif card.id == Fighting_Energy:
                        score += 20   # ルナサイクル・ソルロック攻撃に必要
                    elif card.id in (Boss_Orders, Zeigh, Lillie, Explorer, Poke_Pad):
                        score += 30

                elif context == SelectContext.ATTACH_FROM:
                    # 特殊効果でエネルギーを移動する場合のターゲット選択
                    score = energy_score(None, card, o.area == AreaType.ACTIVE)

                elif context in (SelectContext.TO_BENCH, SelectContext.TO_FIELD):
                    if card.id == Solrock:
                        score = 110
                    elif card.id == Lunatone:
                        score = 105
                    elif card.id == Charcadet:
                        score = 100
                    else:
                        score = 80

                elif context == SelectContext.DISCARD:
                    if o.area == AreaType.DECK:
                        # Perfect Mixer / モグリューのほりまくり: 炎エネ・闘エネをトラッシュ
                        if card.id == Fire_Energy:
                            score = 200
                        elif card.id == Fighting_Energy:
                            score = 200
                        else:
                            score = -200
                    else:
                        is_lunacycle = select.effect is not None and select.effect.id == Lunatone
                        if is_lunacycle:
                            # ルナサイクル: 闘エネルギーをトラッシュ
                            score = 200 if card.id == Fighting_Energy else -100
                        elif card.id == Fire_Energy:
                            score = 200
                        elif card.id == Fighting_Energy:
                            score = 180  # 炎エネに次いで捨てても良い
                        elif card.id in (Charcadet, Soubureizu_ex, Solrock, Lunatone):
                            score = -100
                        elif card.id in (Explorer, Night_Stretcher):
                            score = -50
                        elif card.id == Boss_Orders:
                            score = -30
                        elif card.id in (Lillie, Zeigh):
                            # ベンチ未展開またはエネルギーが手札にない場合はドロー系サポートを温存
                            score = -100 if need_draw else 50
                        else:
                            score = 50

                elif context == SelectContext.EFFECT_TARGET:
                    if isinstance(card, Pokemon) and o.playerIndex != my_index:
                        data = card_table[card.id]
                        if data.ex:
                            dmg   = sinhomura_damage(trash_fire, card)
                            score = 1000 if card.hp <= dmg else 500 + dmg
                        elif card.id in op_attacker_ids and len(card.energies) >= 1:
                            dmg   = sinhomura_damage(trash_fire, card)
                            score = 900 if card.hp <= dmg else 400 + dmg
                        else:
                            score = 100

                elif context in (SelectContext.TO_DECK, SelectContext.TO_DECK_ENERGY):
                    score = -100 if card.id in (Fire_Energy, Fighting_Energy) else 100

                elif context in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
                    score = 100

        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table[card.id]
            if data.cardType == CardType.POKEMON:
                score = 20000
                total_fighters = (field_counts[Solrock] + field_counts[Lunatone] +
                                  field_counts[Charcadet] + field_counts[Soubureizu_ex])
                bench_full = len(my_state.bench) >= my_state.benchMax
                if bench_full or total_fighters >= 5:
                    score = -1
            else:
                score = 10000
                if card.id == Perfect_Mixer:
                    score = 9900 if my_state.deckCount >= 1 else -1

                elif card.id == Hyperball:
                    need_soubureizu = (field_counts[Soubureizu_ex] < 1 or
                                       (trash_fire >= SWITCH_FIRE_THRESHOLD and field_counts[Soubureizu_ex] < 2))
                    need_solrock    = (field_counts[Solrock] + hand_counts[Solrock]) < 1
                    need_lunatone   = (field_counts[Lunatone] + hand_counts[Lunatone]) < 1
                    if len(my_state.hand) < 3:
                        score = -1
                    elif need_soubureizu:
                        score = 8500
                    elif need_solrock or need_lunatone:
                        score = 7000
                    else:
                        score = -1

                elif card.id == Fight_Gong:
                    # 闘タイプたねポケモンまたは基本闘エネルギーをサーチ
                    solrock_avail  = (field_counts[Solrock]  + hand_counts[Solrock])  >= 1
                    lunatone_avail = (field_counts[Lunatone] + hand_counts[Lunatone]) >= 1
                    if not solrock_avail:
                        score = 9000
                    elif not lunatone_avail:
                        score = 8500
                    elif hand_counts[Fighting_Energy] < 2:
                        score = 7000  # ルナサイクル用の闘エネを補充
                    else:
                        score = 5000

                elif card.id == Poke_Pad:
                    need_solrock  = (field_counts[Solrock]  + hand_counts[Solrock])  < 1
                    need_lunatone = (field_counts[Lunatone] + hand_counts[Lunatone]) < 1
                    need_charcadet = ((field_counts[Charcadet] + field_counts[Soubureizu_ex]
                                       + hand_counts[Charcadet]) < 1)
                    if need_solrock:
                        score = 8500
                    elif need_lunatone:
                        score = 8000
                    elif need_charcadet:
                        score = 7000
                    else:
                        score = -1

                elif card.id == Night_Stretcher:
                    has_pokemon_trash = any(c.id in (Charcadet, Soubureizu_ex, Solrock, Lunatone)
                                            for c in my_state.discard)
                    active_soubureizu_no_fire = (
                        my_active is not None and
                        my_active.id == Soubureizu_ex and
                        sum(1 for e in my_active.energies if e == EnergyType.FIRE) == 0 and
                        discard_counts[Fire_Energy] >= 1
                    )
                    if active_soubureizu_no_fire:
                        score = 8900  # 攻撃を通すためにトラッシュから炎エネを最優先回収
                    elif has_pokemon_trash:
                        score = 7000
                    else:
                        score = -1

                elif card.id == Boss_Orders:
                    score = 3200 if plan.target >= 1 else -1

                elif card.id == Explorer:
                    # 上6枚を見て4枚トラッシュ+2枚取得 → 残り3枚確保: deck < 9 なら禁止
                    if my_state.deckCount < 9:
                        score = -1
                    else:
                        score = 3100

                elif card.id == Lillie:
                    # 手札を山札に戻してから6枚引く → 残り3枚確保: (deck + hand) < 9 なら禁止
                    if (my_state.deckCount + len(my_state.hand)) < 9:
                        score = -1
                    elif len(my_state.hand) <= 4:
                        score = 3000
                    else:
                        score = 1000

                elif card.id == Zeigh:
                    is_first_turn = state.turn == 1 and state.firstPlayer == my_index
                    # 5枚引く → 残り3枚確保: deck < 8 なら禁止
                    if my_state.deckCount < 8:
                        score = -1
                    elif is_first_turn:
                        score = 3500
                    elif hand_counts[Fighting_Energy] >= 3:
                        score = 2500  # 闘エネを大量にトラッシュできる
                    elif hand_counts[Fire_Energy] >= 3:
                        score = 2500  # 炎エネも同様
                    elif len(my_state.hand) >= 6:
                        score = 2000
                    else:
                        score = 1500

        elif o.type == OptionType.ATTACH:
            energy_card = get_card(obs, o.area, o.index, my_index)
            pokemon     = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            eid         = energy_card.id if energy_card else None
            score       = energy_score(eid, pokemon, o.inPlayArea == AreaType.ACTIVE)
            if o.inPlayArea == AreaType.ACTIVE and plan.attacker == 0 and plan.energy:
                score += 200
            elif o.inPlayArea == AreaType.BENCH and plan.attacker == 1 + o.inPlayIndex and plan.energy:
                score += 200

        elif o.type == OptionType.EVOLVE:
            pokemon  = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            evo_card = get_card(obs, AreaType.HAND, o.index, my_index)
            if evo_card is not None and evo_card.id == Soubureizu_ex:
                score = 10000 + len(pokemon.energies)  # ソウブレイズexへの進化は最高優先
            else:
                score = 9000 + len(pokemon.energies)

        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card is not None and card.id == Lunatone:
                # ルナサイクル: ソルロックがいて闘エネを手札から捨てて3枚引く
                solrock_on_field = any(
                    p is not None and p.id == Solrock
                    for p in ([my_active] if my_active else []) + list(my_state.bench)
                )
                if solrock_on_field and hand_counts[Fighting_Energy] >= 1:
                    if my_state.deckCount < 6:  # 3枚引く → 残り3枚以上確保
                        score = -1
                    else:
                        score = 9800
                else:
                    score = -1
            elif card is not None and card.id in (Mogurew, Kitchigisu_ex):
                score = 8000  # ほりまくり / さかてにとる: 常に使用
            else:
                score = 500

        elif o.type == OptionType.RETREAT:
            active_id        = my_active.id if my_active else -1
            non_attacker_ids = {Charcadet, Lunatone, Mogurew, Kitchigisu_ex}
            preferred_on_bench = any(
                p is not None and p.id == preferred_attacker for p in my_state.bench
            )
            active_is_non_attacker   = active_id in non_attacker_ids
            active_is_wrong_attacker = (active_id == Solrock and preferred_attacker == Soubureizu_ex and
                                        any(p is not None and p.id == Soubureizu_ex for p in my_state.bench))

            if (active_is_non_attacker or active_is_wrong_attacker) and preferred_on_bench:
                score = 9800  # アイテムより高くして即座に退場
            elif plan.attacker >= 1:
                score = 2000
            else:
                score = -1

        elif o.type == OptionType.ATTACK:
            score = 1000
            if o.attackId in (ATK_SINHOMURA, ATK_COSMOBEAM):
                score += 100  # 使用したい技に加点

        scores.append(score)

    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    return desc_indices[:select.maxCount]
