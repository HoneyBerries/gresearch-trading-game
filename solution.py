import math
import random

# Round 3 strategy: rewards are shared, and if two or more teams accept on the
# same turn they all score zero (and are still locked out). So the value of
# accepting a reward r is r * P(nobody else accepts), not r.
#
# We model every other team from what we can observe:
#   - a score jump of size x means they took the reward x (exact accept round)
#   - an unlocked -> locked flip means they accepted (possibly a collision)
#   - turns where they were free and did not lock mean they passed that reward
# From this we predict who is free this turn and how likely each free team is to
# accept the current reward, and accept when r * P(no collision) beats the
# opportunity cost of sitting out the lockout.

PRIOR_G = 1.0          # prior points-per-round used for the opportunity cost
PRIOR_WEIGHT = 150.0   # rounds of pseudo-evidence behind PRIOR_G
PRIOR_CENTER = 18.0    # prior acceptance curve for unknown teams (logistic)
PRIOR_SCALE = 5.0
PRIOR_STRENGTH = 1.0   # pseudo-observations behind the prior curve
DECAY = 0.997          # recency weighting of observations per round
MAX_OBS = 400          # per-team observations kept


class Team:
	def __init__(self):
		self.prev_score = None
		self.prev_locked = None
		self.accepts = set()        # rounds this team accepted on
		self.last_acc = -10 ** 9    # most recent accept round
		self.finalized = 1          # next round to turn into an observation
		self.obs = []               # (round, reward, accepted)


class Strategy:
	def __init__(self):
		self.reset()

	def reset(self):
		self.history = {}           # history[k] = reward on round k
		self.last_t = 0
		self.teams = []
		self.offset_votes = {0: 0, 1: 0, 2: 0}
		self.my_lock_start = None
		self.lock_len = None
		self.jitter = random.uniform(0.9, 1.1)  # breaks symmetry with clones

	def offset(self):
		# Rounds between an accept and the first observation showing the lock.
		best = max(self.offset_votes, key=lambda k: self.offset_votes[k])
		return best if self.offset_votes[best] > 0 else 0

	def match_teams(self, t, other_scores):
		# The list order is not guaranteed (it may be sorted by score), so pair
		# each entry with the team whose previous score explains it: unchanged,
		# or increased by last round's reward. Ties prefer same lock state, then
		# same position.
		r_prev = self.history.get(t - 1)
		free = list(range(len(self.teams)))
		order = [None] * len(other_scores)
		for want_jump in (True, False):
			for j, (score, locked) in enumerate(other_scores):
				if order[j] is not None:
					continue
				best, best_key = None, None
				for i in free:
					prev = self.teams[i].prev_score
					if prev is None:
						ok = not want_jump
					elif want_jump:
						ok = r_prev is not None and abs(score - prev - r_prev) < 1e-3
					else:
						ok = abs(score - prev) < 1e-3
					if ok:
						key = (self.teams[i].prev_locked != locked, i != j)
						if best_key is None or key < best_key:
							best, best_key = i, key
				if best is not None:
					order[j] = best
					free.remove(best)
		for j in range(len(order)):
			if order[j] is None:
				order[j] = free.pop(0) if free else None
		if None in order:
			return [Team() for _ in other_scores]
		return [self.teams[i] for i in order]

	def update_teams(self, t, other_scores, lockout):
		if len(other_scores) != len(self.teams):
			self.teams = [Team() for _ in other_scores]
		else:
			self.teams = self.match_teams(t, other_scores)
		obs_round = t - 1  # other_scores describe the state after round t-1
		L = self.lock_len or lockout
		for team, (score, locked) in zip(self.teams, other_scores):
			if team.prev_score is not None:
				delta = score - team.prev_score
				acc_round = None
				if abs(delta) > 1e-9:
					for k in range(obs_round, max(0, obs_round - 3), -1):
						if abs(self.history.get(k, 0.0) - delta) < 1e-3:
							acc_round = k
							break
				flipped = locked and not team.prev_locked
				if acc_round is not None:
					if flipped:
						self.offset_votes[min(2, obs_round - acc_round)] += 1
				elif flipped:
					acc_round = obs_round - self.offset()
				elif locked and team.prev_locked and team.last_acc > 0 and obs_round > team.last_acc + L:
					# Still locked after the old lockout ran out: they re-accepted
					# straight away and the flip was hidden by the reporting lag.
					acc_round = obs_round
				if acc_round is not None and acc_round > team.last_acc + L - 2:
					team.accepts.add(acc_round)
					team.last_acc = max(team.last_acc, acc_round)
			team.prev_score = score
			team.prev_locked = locked

			# Turn rounds that are now settled into accept / pass observations.
			team.finalized = max(team.finalized, t - 50)  # skip gaps in t
			while team.finalized <= t - 3:
				k = team.finalized
				if k in team.accepts:
					team.obs.append((k, self.history.get(k, 0.0), True))
				else:
					prior_acc = max((a for a in team.accepts if a < k), default=-10 ** 9)
					if k > prior_acc + L:
						team.obs.append((k, self.history.get(k, 0.0), False))
				team.finalized += 1
			if len(team.obs) > MAX_OBS:
				del team.obs[: len(team.obs) - MAX_OBS]

	def free_prob(self, team, t, lockout):
		L = self.lock_len or lockout
		if team.prev_locked:
			if team.last_acc < 0:
				return 0.5
			if team.last_acc + L < t:
				return 1.0
			if team.last_acc + L == t:
				return 0.5  # one-turn uncertainty around unlock
			return 0.0
		return 1.0

	def accept_prob(self, team, r, t):
		# Threshold-style evidence: accepting at a <= r means they would accept r,
		# passing at b >= r means they would pass r.
		p0 = 1.0 / (1.0 + math.exp(-(r - PRIOR_CENTER) / PRIOR_SCALE))
		yes = PRIOR_STRENGTH * p0
		no = PRIOR_STRENGTH * (1.0 - p0)
		for k, x, acc in team.obs:
			w = DECAY ** (t - k)
			if acc and x <= r:
				yes += w
			elif not acc and x >= r:
				no += w
		return yes / (yes + no)

	def play(self, reward, lockout, t, T, your_score, other_scores, is_locked):
		if t == 1 or t <= self.last_t:
			self.reset()
		self.history[t] = reward
		self.last_t = t

		# Learn our own lockout length from experience.
		if self.my_lock_start is not None and not is_locked:
			self.lock_len = t - self.my_lock_start - 1
			self.my_lock_start = None

		self.update_teams(t, other_scores, lockout)

		if is_locked:
			return False

		p_clear = 1.0
		for team in self.teams:
			f = self.free_prob(team, t, lockout)
			if f > 0.0:
				p_clear *= 1.0 - f * self.accept_prob(team, reward, t)

		g = (PRIOR_G * PRIOR_WEIGHT + your_score) / (PRIOR_WEIGHT + t - 1)
		L = self.lock_len or lockout
		cost = g * min(L, T - t) * self.jitter

		accept = reward * p_clear > cost and reward > 0
		if accept:
			self.my_lock_start = t
		return bool(accept)


play = Strategy().play
