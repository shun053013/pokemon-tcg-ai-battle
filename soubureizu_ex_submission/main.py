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
Soubureizu ex Deck
This deck burns through Fire Energy in the discard pile to power up しんえんほむら.
"""

# Load deck.csv in the dataset
file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path
with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = []
for i in range(60):
    my_deck.append(int(csv[i]))

# Fetch card metadata database and create an ID-to-Card lookup table
all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

# Decklist
Charcadet       = 319   # カルボウ
Soubureizu_ex   = 320   # ソウブレイズex
Fire_Energy     = 2     # 基本炎エネルギー
Mist_Energy     = 11    # ミストエネルギー（無色+相手ワザ効果無効）

Hyperball       = 1121  # ハイパーボール
Poke_Pad        = 1152  # ポケパッド
Night_Stretcher = 1097  # 夜のタンカ
Poke_Poffin     = 1086  # なかよしポフィン
Perfect_Mixer   = 1128  # パーフェクトミキサー

Lillie          = 1227  # リーリエの決心
Zeigh           = 1192  # ゼイユ
Explorer        = 1185  # 探検家の先導
Judgeman        = 1213  # ジャッジマン（おたがい手札を戻して4枚引く）

# ケーシィ・ユンゲラー・フーディン（ミストエネ最優先付与のトリガー）
ABRA_ALAKAZAM_LINE = {109, 741, 742, 245, 743}

TOTAL_FIRE = 19  # デッキ内の炎エネルギー総数

# Precompute attack IDs for Soubureizu ex
_atk_id = {a.name: a.attackId for a in all_attack()}
ATK_SINHOMURA = _atk_id.get("しんえんほむら", -1)   # 1 fire, 30 + 20×trash fire


class AttackPlan:
    attacker     = -1   # index in my_cards (0=active, 1+=bench)
    target       = -1   # index in op_cards
    attack_index = -1   # 0=しんえんほむら, 1=アメジストレイジ
    remain_hp    = -1
    energy       = False  # True if energy attachment is still needed


plan             = AttackPlan()
pre_turn         = 0
op_attacker_ids: set = set()  # 対戦中に相手アクティブに出てきたポケモンのカードID


def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    """Helper function to safely extract a Card or Pokemon object from specific zones."""
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


def prize_count(pokemon: Pokemon) -> int:
    """Calculates how many Prize cards a Pokémon yields upon being Knocked Out."""
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == 12:  # Legacy Energy
            count -= 1
    return max(0, count)


def sinhomura_damage(trash_fire: int, target: Pokemon) -> int:
    damage = 30 + trash_fire * 20
    data = card_table[target.id]
    if data.weakness == EnergyType.FIRE:
        damage *= 2
    return damage


def agent(obs_dict: dict) -> list[int]:
    """Main Agent Function.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.

    Returns:
        list[int]: A list of option index.
    """
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return my_deck

    state   = obs.current
    select  = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    global plan, pre_turn, op_attacker_ids
    if pre_turn != state.turn:
        pre_turn = state.turn
        plan     = AttackPlan()
        # 相手アクティブのポケモンをアタッカー履歴に記録（エネルギーを持っていれば攻撃可能と判断）
        if op_state.active and op_state.active[0] is not None and len(op_state.active[0].energies) >= 1:
            op_attacker_ids.add(op_state.active[0].id)

    field_counts   = defaultdict(int)
    hand_counts    = defaultdict(int)
    discard_counts = defaultdict(int)

    for card in my_state.active + my_state.bench:
        if card is None:
            continue
        field_counts[card.id] += 1

    for card in my_state.hand:
        hand_counts[card.id] += 1

    for card in my_state.discard:
        discard_counts[card.id] += 1

    trash_fire = discard_counts[Fire_Energy]

    # ドロー用サポートをトラッシュしないべき状況：ベンチ未展開 or 炎エネが手札にない
    bench_not_setup = (field_counts[Charcadet] + field_counts[Soubureizu_ex]) < 3
    need_draw = bench_not_setup or hand_counts[Fire_Energy] == 0

    # 山札切れ防止: 残り4枚以下ではドロー・サーチ系カードを使用しない
    deck_low = my_state.deckCount <= 4

    my_active  = my_state.active[0] if my_state.active else None
    op_active  = op_state.active[0] if op_state.active else None
    my_cards   = ([my_active] if my_active is not None else []) + [p for p in my_state.bench if p is not None]
    op_cards   = ([op_active] if op_active is not None else []) + [p for p in op_state.bench  if p is not None]

    # 相手の場・ベンチにケーシィ・ユンゲラー・フーディンがいるか
    op_has_abra_line = any(p is not None and p.id in ABRA_ALAKAZAM_LINE for p in op_cards)

    # 手札+山札にアクセスできる炎エネルギー枚数（0になるとしんえんほむら不可）
    attached_fire   = sum(sum(1 for e in p.energies if e == EnergyType.FIRE) for p in my_cards)
    fire_in_deck    = max(0, TOTAL_FIRE - discard_counts[Fire_Energy]
                             - hand_counts[Fire_Energy] - attached_fire)
    fire_accessible = hand_counts[Fire_Energy] + fire_in_deck

    # Compute attack plan in MAIN context
    if context == SelectContext.MAIN and state.turn >= 2:
        can_switch    = False
        can_op_switch = False
        for o in select.option:
            if o.type == OptionType.RETREAT:
                can_switch = True

        best_score = -1
        for i, my_pokemon in enumerate(my_cards):
            if i != 0 and not can_switch:
                continue
            if my_pokemon.id != Soubureizu_ex:
                continue

            energy_count = len(my_pokemon.energies)

            energy_required = 1
            damage_fn = lambda t: sinhomura_damage(trash_fire, t)

            ec = energy_count
            more_energy = False
            if ec < energy_required:
                can_get_fire = (
                    hand_counts[Fire_Energy] >= 1 or
                    (hand_counts[Night_Stretcher] >= 1 and discard_counts[Fire_Energy] >= 1)
                )
                if can_get_fire and not state.energyAttached:
                    ec += 1
                    more_energy = True
                else:
                    continue

            for j, op_pokemon in enumerate(op_cards):
                if j != 0 and not can_op_switch:
                    break
                damage = damage_fn(op_pokemon)
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

        # バトル場がOHKOできない場合、エネルギー持ちの非exアタッカーをベンチから呼び出す
        if (can_op_switch
                and plan.attacker >= 0
                and plan.target == 0
                and plan.remain_hp > 0):
            my_attacker = my_cards[plan.attacker] if plan.attacker < len(my_cards) else None
            ec_now = len(my_attacker.energies) if my_attacker else 0
            can_still_attack = ec_now >= 1 or (hand_counts[Fire_Energy] >= 1 and not state.energyAttached)
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
                        continue  # exは上のループで処理済み
                    damage = sinhomura_damage(trash_fire, op_pokemon)
                    score  = int(1000 * damage / op_pokemon.hp) + len(op_pokemon.energies) * 100 + 600
                    if op_pokemon.hp <= damage:
                        score += 2000
                    if score > best_score:
                        best_score        = score
                        plan.attacker     = 0
                        plan.target       = j
                        plan.attack_index = 0
                        plan.remain_hp    = op_pokemon.hp - damage
                        plan.energy       = ec_now < 1

    # Energy attachment priority score
    def energy_score(energy_id: int | None, pokemon: Pokemon, active: bool) -> int:
        count = len(pokemon.energies)
        soubureizu_on_bench = any(p is not None and p.id == Soubureizu_ex for p in my_state.bench)
        active_is_charcadet = my_active is not None and my_active.id == Charcadet

        # アクティブがカルボウ＋ソウブレイズexがベンチにいる場合：
        # カルボウにエネルギーをつけて逃げられるようにする（進化できない場合優先）
        if pokemon.id == Charcadet and active and active_is_charcadet and soubureizu_on_bench and count == 0:
            can_evolve = hand_counts[Soubureizu_ex] > 0
            return 8800 if not can_evolve else 3000

        # ソウブレイズex
        if pokemon.id == Soubureizu_ex:
            fire_count = sum(1 for e in pokemon.energies if e == EnergyType.FIRE)
            mist_count = sum(1 for c in pokemon.energyCards if c.id == Mist_Energy)

            # ミストエネルギー: 相手にケーシィラインがいれば最優先で付ける
            if energy_id == Mist_Energy and mist_count == 0 and op_has_abra_line:
                # 炎エネ付き（攻撃可能）→ 最高優先、炎エネなし→ 炎エネ付与(8010)より低め
                return 10500 if fire_count >= 1 else 7900

            if fire_count >= 1:
                return 50   # 炎エネあり、追加不要
            if energy_id == Fire_Energy:
                return 8010 + (10 if active else 0)
            return 200

        if pokemon.id == Charcadet:
            return 1000
        return 500

    # Iterate over every possible option and assign a heuristic score
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
                        if card.id == Soubureizu_ex:
                            score = 100 + energy_count
                        elif card.id == Charcadet:
                            score = 50 + energy_count
                        if plan.attacker > 0 and o.area == AreaType.BENCH and o.index == plan.attacker - 1:
                            score += 200
                    else:
                        if plan.target > 0 and o.index == plan.target - 1:
                            score += 200

                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    if card.id == Charcadet:
                        score = 10
                    elif card.id == Soubureizu_ex:
                        score = 5

                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    score = 10 if card.id == Charcadet else 1

                elif context == SelectContext.TO_HAND:
                    score = 200 - hand_counts[card.id] * 50
                    if card.id == Charcadet:
                        score += 50 if field_counts[Charcadet] + field_counts[Soubureizu_ex] < 2 else -50
                    elif card.id == Soubureizu_ex:
                        score += 80 if field_counts[Soubureizu_ex] < 1 else -100
                    elif card.id == Fire_Energy:
                        is_tanker = select.effect is not None and select.effect.id == Night_Stretcher
                        soubureizu_needs_fire = (
                            is_tanker and field_counts[Soubureizu_ex] >= 1 and all(
                                sum(1 for e in p.energies if e == EnergyType.FIRE) == 0
                                for p in my_cards if p.id == Soubureizu_ex
                            )
                        )
                        if soubureizu_needs_fire:
                            score += 300  # 夜のタンカで攻撃用炎エネを優先取得
                        else:
                            score -= 100  # 通常はトラッシュに残す
                    elif card.id == Mist_Energy:
                        score += 50 if op_has_abra_line else -50
                    elif card.id in (Zeigh, Lillie, Explorer, Poke_Pad, Judgeman):
                        score += 30

                elif context == SelectContext.ATTACH_FROM:
                    score = energy_score(None, card, o.area == AreaType.ACTIVE)

                elif context in (SelectContext.TO_BENCH, SelectContext.TO_FIELD):
                    score = 100 if card.id == Charcadet else 80 if card.id == Soubureizu_ex else 10

                elif context == SelectContext.DISCARD:
                    if o.area == AreaType.DECK:
                        # Perfect Mixer: 炎エネ優先（残2以下は保護）、ミストエネは保護
                        if card.id == Fire_Energy:
                            score = 200 if fire_accessible >= 3 else -200
                        else:
                            score = -200
                    else:
                        if card.id == Fire_Energy:
                            score = 150 if fire_accessible >= 2 else -200
                        elif card.id == Mist_Energy:
                            score = -150  # ミストエネはコストにしない
                        elif card.id in (Charcadet, Soubureizu_ex):
                            score = -100
                        elif card.id in (Explorer, Night_Stretcher):
                            score = -50
                        elif card.id in (Lillie, Zeigh, Judgeman):
                            score = -100 if need_draw else 50
                        else:
                            score = 50

                elif context == SelectContext.EFFECT_TARGET:
                    # Boss Orders: exまたは既知の非exアタッカーをベンチから呼び出す
                    if isinstance(card, Pokemon) and o.playerIndex != my_index:
                        data = card_table[card.id]
                        if data.ex:
                            dmg = sinhomura_damage(trash_fire, card)
                            score = 1000 if card.hp <= dmg else 500 + dmg
                        elif card.id in op_attacker_ids and len(card.energies) >= 1:
                            # 対戦中に攻撃してきた非exアタッカー（エネルギー持ち）
                            dmg = sinhomura_damage(trash_fire, card)
                            score = 900 if card.hp <= dmg else 400 + dmg
                        else:
                            score = 100

                elif context in (SelectContext.TO_DECK, SelectContext.TO_DECK_ENERGY):
                    score = -100 if card.id == Fire_Energy else 100

                elif context in (SelectContext.DISCARD_ENERGY, SelectContext.DISCARD_ENERGY_CARD):
                    # 逃げるコスト: 炎エネ最優先（枚数が多いためトラッシュに送ってOK）
                    if card.id == Fire_Energy:
                        score = 200
                    else:
                        score = 100

                elif context in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
                    score = 100

        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table[card.id]
            if data.cardType == CardType.POKEMON:
                score = 20000
                charcadet_field = field_counts[Charcadet] + field_counts[Soubureizu_ex]
                bench_full = len(my_state.bench) >= my_state.benchMax
                if bench_full or charcadet_field >= 4:
                    score = -1
            else:
                score = 10000
                if card.id == Poke_Poffin:
                    charcadet_field = field_counts[Charcadet] + field_counts[Soubureizu_ex]
                    bench_full = len(my_state.bench) >= my_state.benchMax
                    if deck_low:
                        score = -1
                    else:
                        score = 9500 if not bench_full and charcadet_field < 4 else -1
                elif card.id == Hyperball:
                    if deck_low:
                        score = -1
                    elif field_counts[Soubureizu_ex] >= 1:
                        score = -1  # 1体でもいれば無理して使わない
                    elif len(my_state.hand) >= 3:
                        score = 8500
                    else:
                        score = -1
                elif card.id == Poke_Pad:
                    if deck_low:
                        score = -1
                    else:
                        score = 8000 if not (field_counts[Charcadet] >= 1 or hand_counts[Charcadet] >= 1) else -1
                elif card.id == Night_Stretcher:
                    # 夜のタンカはトラッシュ回収なので山札を消費しない → deck_low 制限なし
                    has_pokemon_trash = any(c.id in (Charcadet, Soubureizu_ex) for c in my_state.discard)
                    active_soubureizu_no_fire = (
                        my_active is not None and
                        my_active.id == Soubureizu_ex and
                        sum(1 for e in my_active.energies if e == EnergyType.FIRE) == 0 and
                        hand_counts[Fire_Energy] == 0 and  # 手札に炎エネがない場合のみ（あれば直接つければいい）
                        discard_counts[Fire_Energy] >= 1
                    )
                    if active_soubureizu_no_fire:
                        score = 8900
                    elif has_pokemon_trash:
                        score = 7000
                    else:
                        score = -1
                elif card.id == Perfect_Mixer:
                    score = -1 if deck_low else (9900 if my_state.deckCount >= 1 else -1)
                elif card.id == Explorer:
                    score = -1 if deck_low else 3100
                elif card.id == Judgeman:
                    if deck_low:
                        score = -1
                    elif op_state.handCount > 8:
                        score = 3300
                    else:
                        score = 2000
                elif card.id == Lillie:
                    if deck_low:
                        score = -1
                    else:
                        score = 3000 if len(my_state.hand) <= 4 else 1000
                elif card.id == Zeigh:
                    is_first_turn = state.turn == 1 and state.firstPlayer == my_index
                    if deck_low:
                        score = -1
                    elif is_first_turn:
                        score = 3500
                    elif hand_counts[Fire_Energy] >= 3:
                        score = 2500
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

        elif o.type == OptionType.RETREAT:
            active_is_charcadet = my_active is not None and my_active.id == Charcadet
            soubureizu_on_bench = any(p is not None and p.id == Soubureizu_ex for p in my_state.bench)
            if active_is_charcadet and soubureizu_on_bench:
                score = 9800  # アイテム(最大9500)より高くして即座に退場を確定
            elif plan.attacker >= 1:
                score = 2000
            else:
                score = -1

        elif o.type == OptionType.ATTACK:
            score = 1000
            if o.attackId == ATK_SINHOMURA:
                score += 100

        scores.append(score)

    # Select in descending order of score
    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    return desc_indices[:select.maxCount]
