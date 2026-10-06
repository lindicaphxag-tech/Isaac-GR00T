import Std

namespace SemRepair

/-- A terminal hypothesis set is safe when every surviving world authorizes
    the same exact repair authority.  The surviving set is represented as a
    predicate so the formal core stays dependency-free. -/
def AuthorityHomogeneous {H A : Type}
    (authority : H → A) (survives : H → Prop) : Prop :=
  ∀ ⦃x y⦄, survives x → survives y → authority x = authority y

/-- Full hidden-identity resolution means at most one world survives. -/
def IdentityResolved {H : Type} (survives : H → Prop) : Prop :=
  ∀ ⦃x y⦄, survives x → survives y → x = y

/-- Full identity resolution is always sufficient for repair authority. -/
theorem identityResolved_implies_authorityHomogeneous
    {H A : Type} (authority : H → A) (survives : H → Prop)
    (h : IdentityResolved survives) :
    AuthorityHomogeneous authority survives := by
  intro x y hx hy
  have hxy : x = y := h hx hy
  subst y
  rfl

/-- If every hidden identity has a distinct authority, authority-safe stopping
    collapses to full identity resolution. -/
theorem injective_authority_homogeneous_implies_identityResolved
    {H A : Type} (authority : H → A) (survives : H → Prop)
    (hInjective : Function.Injective authority)
    (hSafe : AuthorityHomogeneous authority survives) :
    IdentityResolved survives := by
  intro x y hx hy
  exact hInjective (hSafe hx hy)

/-- Under an injective authority map, the two stopping criteria are exactly
    equivalent. -/
theorem injective_authority_safeStop_iff_identityResolved
    {H A : Type} (authority : H → A) (survives : H → Prop)
    (hInjective : Function.Injective authority) :
    AuthorityHomogeneous authority survives ↔ IdentityResolved survives := by
  constructor
  · intro hSafe
    exact injective_authority_homogeneous_implies_identityResolved
      authority survives hInjective hSafe
  · intro hIdentity
    exact identityResolved_implies_authorityHomogeneous
      authority survives hIdentity

/-- If all surviving semantic worlds bind the same exact authority, no further
    identity refinement is required by the authority stopping rule. -/
theorem constant_authority_is_homogeneous
    {H A : Type} (authority : H → A) (survives : H → Prop) (a : A)
    (hConstant : ∀ x, authority x = a) :
    AuthorityHomogeneous authority survives := by
  intro x y hx hy
  calc
    authority x = a := hConstant x
    _ = authority y := (hConstant y).symm

/-- Two surviving worlds that require different concrete authorities make a
    terminal stop unsafe, even if some external observer aliases them. -/
theorem conflicting_authorities_block_safe_stop
    {H A : Type} (authority : H → A) (survives : H → Prop)
    {x y : H}
    (hx : survives x) (hy : survives y)
    (hConflict : authority x ≠ authority y) :
    ¬ AuthorityHomogeneous authority survives := by
  intro hSafe
  exact hConflict (hSafe hx hy)

end SemRepair
