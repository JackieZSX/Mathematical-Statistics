import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Convex.Deriv
import Mathlib.Tactic.Linarith

/-!
# Formal Verification of Temperature Scaling MLE & LRT Theory

This file formally proves the core mathematical derivations in:
* Proposition 1: Strict order preservation and argmax invariance of Softmax Temperature Scaling
* First derivative (Score equation): `d/dβ ℓ(β) = U(β) = ∑_i (z_{i, y_i} - E_{p_i(β)}[Z_i])`
* Softmax derivative and second derivative (Observed Fisher Information):
  `d/dβ U(β) = -I_n(β) = -∑_i Var_{p_i(β)}(Z_i)`
* Theorem 1: Strict positivity of Fisher information, strict concavity of `ℓ(β)` on `(0, ∞)`,
  and global uniqueness of the MLE
* Delta method derivative and variance identity for `T = 1 / β`
-/

open Finset Real

namespace TemperatureScaling

section SoftmaxAndOrder

variable {C : ℕ} [NeZero C]

/-- Partition function (normalizing constant) for logits `z` at inverse temperature `β`. -/
noncomputable def denom (z : Fin C → ℝ) (β : ℝ) : ℝ :=
  ∑ c : Fin C, Real.exp (β * z c)

/-- Softmax probability of class `c` at inverse temperature `β`. -/
noncomputable def softmax (z : Fin C → ℝ) (β : ℝ) (c : Fin C) : ℝ :=
  Real.exp (β * z c) / denom z β

lemma denom_pos (z : Fin C → ℝ) (β : ℝ) : 0 < denom z β := by
  unfold denom
  exact Finset.sum_pos (fun c _ ↦ Real.exp_pos (β * z c)) Finset.univ_nonempty

lemma denom_ne_zero (z : Fin C → ℝ) (β : ℝ) : denom z β ≠ 0 :=
  (denom_pos z β).ne'

lemma softmax_pos (z : Fin C → ℝ) (β : ℝ) (c : Fin C) : 0 < softmax z β c :=
  div_pos (Real.exp_pos _) (denom_pos z β)

lemma sum_softmax_eq_one (z : Fin C → ℝ) (β : ℝ) :
    ∑ c : Fin C, softmax z β c = 1 := by
  unfold softmax
  simp only [div_eq_mul_inv, ← Finset.sum_mul]
  exact mul_inv_cancel₀ (denom_ne_zero z β)

/-- **Proposition 1 (Strict Order Preservation)**:
For `β > 0`, `z k < z c ↔ softmax z β k < softmax z β c`. -/
theorem order_preservation (z : Fin C → ℝ) {β : ℝ} (hβ : 0 < β) (c k : Fin C) :
    z k < z c ↔ softmax z β k < softmax z β c := by
  unfold softmax
  have hd : 0 < denom z β := denom_pos z β
  rw [div_lt_div_iff_of_pos_right hd, Real.exp_lt_exp, mul_lt_mul_iff_right₀ hβ]

/-- **Proposition 1 Corollary (Argmax Invariance)**:
For `β > 0`, any maximizer of logits `z` is also a maximizer of `softmax z β`. -/
theorem argmax_invariance (z : Fin C → ℝ) {β : ℝ} (hβ : 0 < β) {c : Fin C}
    (hc : ∀ k : Fin C, z k ≤ z c) :
    ∀ k : Fin C, softmax z β k ≤ softmax z β c := by
  intro k
  unfold softmax
  have hd : 0 < denom z β := denom_pos z β
  rw [div_le_div_iff_of_pos_right hd, Real.exp_le_exp]
  exact mul_le_mul_of_nonneg_left (hc k) hβ.le

end SoftmaxAndOrder

section LikelihoodAndDerivatives

variable {C n : ℕ} [NeZero C]

/-- Conditional expectation `E_{p(β)}[Z] = ∑ c, p_c(β) z_c`. -/
noncomputable def condExpect (z : Fin C → ℝ) (β : ℝ) : ℝ :=
  ∑ c : Fin C, softmax z β c * z c

/-- Conditional variance `Var_{p(β)}(Z) = ∑ c, p_c(β) (z_c - E[Z])^2`. -/
noncomputable def condVar (z : Fin C → ℝ) (β : ℝ) : ℝ :=
  ∑ c : Fin C, softmax z β c * (z c - condExpect z β) ^ 2

/-- Algebraic identity: `∑ p_c z_c^2 - (∑ p_c z_c)^2 = ∑ p_c (z_c - E[Z])^2`. -/
theorem condVar_eq_second_moment_sub_sq (z : Fin C → ℝ) (β : ℝ) :
    (∑ c : Fin C, softmax z β c * z c ^ 2) - (condExpect z β) ^ 2 = condVar z β := by
  unfold condVar
  have hsum : ∑ c : Fin C, softmax z β c = 1 := sum_softmax_eq_one z β
  have hexp : ∑ c : Fin C, softmax z β c * z c = condExpect z β := rfl
  calc
    (∑ c : Fin C, softmax z β c * z c ^ 2) - (condExpect z β) ^ 2
      = (∑ c : Fin C, softmax z β c * z c ^ 2)
          - 2 * condExpect z β * (∑ c : Fin C, softmax z β c * z c)
          + (condExpect z β) ^ 2 * (∑ c : Fin C, softmax z β c) := by
            rw [hexp, hsum]; ring
    _ = ∑ c : Fin C, (softmax z β c * z c ^ 2
          - 2 * condExpect z β * (softmax z β c * z c)
          + (condExpect z β) ^ 2 * softmax z β c) := by
            simp only [Finset.mul_sum, Finset.sum_add_distrib, Finset.sum_sub_distrib]
    _ = ∑ c : Fin C, softmax z β c * (z c - condExpect z β) ^ 2 := by
            refine Finset.sum_congr rfl fun c _ ↦ ?_
            ring

omit [NeZero C] in
/-- Derivative of `denom z β` with respect to `β`. -/
lemma hasDerivAt_denom (z : Fin C → ℝ) (β : ℝ) :
    HasDerivAt (denom z) (∑ c : Fin C, Real.exp (β * z c) * z c) β := by
  unfold denom
  have h : ∀ c ∈ (Finset.univ : Finset (Fin C)),
      HasDerivAt (fun b ↦ Real.exp (b * z c)) (Real.exp (β * z c) * z c) β := by
    intro c _
    have hmul : HasDerivAt (fun b ↦ b * z c) (1 * z c) β :=
      (hasDerivAt_id β).mul_const (z c)
    simpa [one_mul] using hmul.exp
  exact HasDerivAt.fun_sum h

/-- Derivative of `log (denom z β)` equals `condExpect z β`. -/
lemma hasDerivAt_log_denom (z : Fin C → ℝ) (β : ℝ) :
    HasDerivAt (fun b ↦ Real.log (denom z b)) (condExpect z β) β := by
  have hd : denom z β ≠ 0 := denom_ne_zero z β
  have hlog := (hasDerivAt_denom z β).log hd
  convert hlog using 1
  unfold condExpect softmax
  simp only [div_eq_mul_inv, Finset.sum_mul]
  refine Finset.sum_congr rfl fun c _ ↦ ?_
  ring

/-- Single-sample log-likelihood `ℓ_i(β) = β z_{y} - log(∑_c exp(β z_c))`. -/
noncomputable def sampleLogLik (z : Fin C → ℝ) (y : Fin C) (β : ℝ) : ℝ :=
  β * z y - Real.log (denom z β)

/-- Total log-likelihood `ℓ(β) = ∑_i ℓ_i(β)`. -/
noncomputable def totalLogLik (z : Fin n → Fin C → ℝ) (y : Fin n → Fin C) (β : ℝ) : ℝ :=
  ∑ i : Fin n, sampleLogLik (z i) (y i) β

/-- Score function `U(β) = ∑_i (z_{i, y_i} - E_{p_i(β)}[Z_i])`. -/
noncomputable def scoreFun (z : Fin n → Fin C → ℝ) (y : Fin n → Fin C) (β : ℝ) : ℝ :=
  ∑ i : Fin n, (z i (y i) - condExpect (z i) β)

/-- **Score Equation Derivation**:
`d/dβ ℓ(β) = U(β)`. -/
theorem hasDerivAt_totalLogLik (z : Fin n → Fin C → ℝ) (y : Fin n → Fin C) (β : ℝ) :
    HasDerivAt (totalLogLik z y) (scoreFun z y β) β := by
  unfold totalLogLik scoreFun
  have h : ∀ i ∈ (Finset.univ : Finset (Fin n)),
      HasDerivAt (sampleLogLik (z i) (y i)) (z i (y i) - condExpect (z i) β) β := by
    intro i _
    have h1 : HasDerivAt (fun b ↦ b * z i (y i)) (z i (y i)) β := by
      simpa [one_mul] using (hasDerivAt_id β).mul_const (z i (y i))
    exact h1.sub (hasDerivAt_log_denom (z i) β)
  exact HasDerivAt.fun_sum h

/-- Derivative of Softmax probability `p_c(β)` equals `p_c(β) * (z_c - E[Z])`. -/
theorem hasDerivAt_softmax (z : Fin C → ℝ) (β : ℝ) (c : Fin C) :
    HasDerivAt (fun b ↦ softmax z b c) (softmax z β c * (z c - condExpect z β)) β := by
  unfold softmax
  have hnum : HasDerivAt (fun b ↦ Real.exp (b * z c)) (Real.exp (β * z c) * z c) β := by
    have hmul : HasDerivAt (fun b ↦ b * z c) (1 * z c) β :=
      (hasDerivAt_id β).mul_const (z c)
    simpa [one_mul] using hmul.exp
  have hden := hasDerivAt_denom z β
  have hdiv := hnum.div hden (denom_ne_zero z β)
  convert hdiv using 1
  have hce : (∑ k : Fin C, Real.exp (β * z k) * z k) / denom z β = condExpect z β := by
    unfold condExpect softmax
    simp only [div_eq_mul_inv, Finset.sum_mul]
    refine Finset.sum_congr rfl fun k _ ↦ ?_
    ring
  rw [← hce]
  field_simp [denom_ne_zero z β]

/-- Derivative of `condExpect z β` equals `condVar z β`. -/
lemma hasDerivAt_condExpect (z : Fin C → ℝ) (β : ℝ) :
    HasDerivAt (condExpect z) (condVar z β) β := by
  unfold condExpect
  have h : ∀ c ∈ (Finset.univ : Finset (Fin C)),
      HasDerivAt (fun b ↦ softmax z b c * z c)
        (softmax z β c * (z c - condExpect z β) * z c) β := by
    intro c _
    exact (hasDerivAt_softmax z β c).mul_const (z c)
  have hsum := HasDerivAt.fun_sum h
  convert hsum using 1
  rw [← condVar_eq_second_moment_sub_sq z β]
  calc
    (∑ c : Fin C, softmax z β c * z c ^ 2) - (condExpect z β) ^ 2
      = (∑ c : Fin C, softmax z β c * z c ^ 2) -
          condExpect z β * (∑ c : Fin C, softmax z β c * z c) := by
            rw [show (∑ c : Fin C, softmax z β c * z c) = condExpect z β from rfl, sq]
    _ = ∑ c : Fin C, (softmax z β c * z c ^ 2 - condExpect z β * (softmax z β c * z c)) := by
            rw [Finset.mul_sum, Finset.sum_sub_distrib]
    _ = ∑ c : Fin C, softmax z β c * (z c - condExpect z β) * z c := by
            refine Finset.sum_congr rfl fun c _ ↦ ?_
            ring

/-- Observed Fisher Information `I_n(β) = ∑_i Var_{p_i(β)}(Z_i)`. -/
noncomputable def fisherInfo (z : Fin n → Fin C → ℝ) (β : ℝ) : ℝ :=
  ∑ i : Fin n, condVar (z i) β

/-- **Second Derivative / Observed Fisher Information Formula**:
`d/dβ U(β) = - I_n(β)`. -/
theorem hasDerivAt_scoreFun (z : Fin n → Fin C → ℝ) (y : Fin n → Fin C) (β : ℝ) :
    HasDerivAt (scoreFun z y) (- fisherInfo z β) β := by
  unfold scoreFun fisherInfo
  have h : ∀ i ∈ (Finset.univ : Finset (Fin n)),
      HasDerivAt (fun b ↦ z i (y i) - condExpect (z i) b) (- condVar (z i) β) β := by
    intro i _
    have hc : HasDerivAt (fun _ : ℝ ↦ z i (y i)) 0 β := hasDerivAt_const β (z i (y i))
    simpa using hc.fun_sub (hasDerivAt_condExpect (z i) β)
  simpa [Finset.sum_neg_distrib] using HasDerivAt.fun_sum h

end LikelihoodAndDerivatives

section ConcavityAndUniqueness

variable {C n : ℕ} [NeZero C]

lemma condVar_nonneg (z : Fin C → ℝ) (β : ℝ) : 0 ≤ condVar z β := by
  unfold condVar
  exact Finset.sum_nonneg fun c _ ↦ mul_nonneg (softmax_pos z β c).le (sq_nonneg _)

/-- If logits `z` are non-degenerate (`∃ c k, z c ≠ z k`), then `0 < condVar z β`. -/
theorem condVar_pos_of_ne (z : Fin C → ℝ) (β : ℝ) (hne : ∃ c k : Fin C, z c ≠ z k) :
    0 < condVar z β := by
  rcases hne with ⟨c, k, hck⟩
  have hterm : ∃ m : Fin C, 0 < softmax z β m * (z m - condExpect z β) ^ 2 := by
    by_cases hc : z c = condExpect z β
    · refine ⟨k, mul_pos (softmax_pos z β k) (sq_pos_of_ne_zero ?_)⟩
      intro hk
      apply hck
      linarith
    · refine ⟨c, mul_pos (softmax_pos z β c) (sq_pos_of_ne_zero (sub_ne_zero.mpr hc))⟩
  rcases hterm with ⟨m, hm⟩
  unfold condVar
  exact Finset.sum_pos' (fun j _ ↦ mul_nonneg (softmax_pos z β j).le (sq_nonneg _))
    ⟨m, Finset.mem_univ m, hm⟩

/-- **Theorem 1a (Strict Positivity of Observed Fisher Information)**:
If at least one sample has non-degenerate logits, then `I_n(β) > 0` for all `β ∈ ℝ`. -/
theorem fisherInfo_pos (z : Fin n → Fin C → ℝ) (β : ℝ)
    (hnd : ∃ (i : Fin n) (c k : Fin C), z i c ≠ z i k) :
    0 < fisherInfo z β := by
  rcases hnd with ⟨i, c, k, hck⟩
  unfold fisherInfo
  exact Finset.sum_pos' (fun j _ ↦ condVar_nonneg (z j) β)
    ⟨i, Finset.mem_univ i, condVar_pos_of_ne (z i) β ⟨c, k, hck⟩⟩

/-- **Theorem 1b (Strict Concavity of Log-Likelihood)**:
Under non-degeneracy, `totalLogLik z y` is strictly concave on `(0, ∞)`. -/
theorem strictConcaveOn_totalLogLik (z : Fin n → Fin C → ℝ) (y : Fin n → Fin C)
    (hnd : ∃ (i : Fin n) (c k : Fin C), z i c ≠ z i k) :
    StrictConcaveOn ℝ (Set.Ioi 0) (totalLogLik z y) := by
  have h_univ : StrictConcaveOn ℝ Set.univ (totalLogLik z y) := by
    apply strictConcaveOn_of_deriv2_neg convex_univ
    · exact ContinuousOn.congr (f := totalLogLik z y)
        (fun b _ ↦ (hasDerivAt_totalLogLik z y b).continuousAt.continuousWithinAt)
        (fun _ _ ↦ rfl)
    · intro b _
      have h1 : deriv (totalLogLik z y) = scoreFun z y := by
        ext t
        exact (hasDerivAt_totalLogLik z y t).deriv
      rw [Function.iterate_succ_apply', Function.iterate_one, h1]
      have h2 := (hasDerivAt_scoreFun z y b).deriv
      rw [h2]
      exact neg_lt_zero.mpr (fisherInfo_pos z b hnd)
  exact h_univ.subset (Set.subset_univ _) (convex_Ioi 0)

/-- **Theorem 1c (Global Uniqueness of MLE)**:
Any two critical points of `totalLogLik` on `(0, ∞)` (where `U(β) = 0`) are equal. -/
theorem mle_unique_of_score_zero (z : Fin n → Fin C → ℝ) (y : Fin n → Fin C)
    (hnd : ∃ (i : Fin n) (c k : Fin C), z i c ≠ z i k)
    {β₁ β₂ : ℝ} (_hβ₁ : 0 < β₁) (_hβ₂ : 0 < β₂)
    (h1 : scoreFun z y β₁ = 0) (h2 : scoreFun z y β₂ = 0) :
    β₁ = β₂ := by
  have hsm : StrictMono (fun b ↦ -scoreFun z y b) := by
    apply strictMono_of_deriv_pos
    intro b
    have hd : HasDerivAt (fun t ↦ -scoreFun z y t) (fisherInfo z b) b := by
      have hneg := (hasDerivAt_scoreFun z y b).fun_neg
      simpa using hneg
    rw [hd.deriv]
    exact fisherInfo_pos z b hnd
  rcases lt_trichotomy β₁ β₂ with hlt | heq | hgt
  · have := hsm hlt
    linarith
  · exact heq
  · have := hsm hgt
    linarith

end ConcavityAndUniqueness

section DeltaMethod

/-- **Delta Method Derivative & Variance Formula**:
For `g(β) = 1 / β`, `g'(β) = -1 / β^2`, and `[g'(β)]^2 / I = T^4 / I` where `T = 1 / β`. -/
theorem hasDerivAt_inv_temp {β : ℝ} (hβ : β ≠ 0) :
    HasDerivAt (fun b : ℝ ↦ 1 / b) (-1 / β ^ 2) β := by
  convert hasDerivAt_inv hβ using 1
  · ext b; simp [one_div]
  · ring

theorem delta_variance_identity {β I : ℝ} (hβ : β ≠ 0) :
    (-1 / β ^ 2) ^ 2 / I = (1 / β) ^ 4 / I := by
  field_simp [hβ]

end DeltaMethod

end TemperatureScaling
