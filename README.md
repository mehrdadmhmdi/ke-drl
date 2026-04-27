# ke-drl
Offline Multi-Dimensional Distributional RL - RKHS Mean Embedding Estimation

## Installation
### Works on Linux/macOS/Windows (requires GPU, Python ≥3.9, git, and pip). 
```python
# install directly from GitHub
python -m pip install "git+https://github.com/mehrdadmhmdi/ke-drl.git"

```

### Developer install (locally editable)

```python
# clone the repo
git clone https://github.com/mehrdadmhmdi/ke-drl.git
cd ke-drl

# install in editable mode
python -m pip install -e .
```

<h3>The KE-DRL Algorithm</h3>

<hr>

<ol>
<li>
<b>Input:</b> Historical data 
<code>𝒟 = {hᵢ}ᵢ₌₁ᴺ = {{(sᵢₜ, aᵢₜ, rᵢₜ)}ₜ₌₁ᵀⁱ}ᵢ₌₁ᴺ</code>, 
regularization <code>λ_reg</code>, discount factor <code>γ</code>, 
Matérn parameters <code>(ν, ℓ)</code>, target policy <code>π</code>, 
evaluation pair <code>(s*, a*)</code>, grid <code>Z^π</code>, 
initialization <code>B_init</code>, fixed-point penalty <code>λ_FP</code>, 
mass-anchor penalty <code>λ_mass</code>.
</li>

<li><b>Pre-computations:</b></li>
<ul>
<li>Compute Gram matrices <code>K̃_(s,a)</code>, <code>K̃_(s′,a′)</code>, <code>K_Zπ</code>.</li>
<li>Compute kernel vector <code>k̃_(s,a)</code> using fixed <code>(s*, a*)</code>.</li>
<li>Compute importance weights <code>α̂</code>.</li>
</ul>

<li><b>Auxiliary Operators:</b></li>
<ul>
<li><code>Γ_(s,a) ← Γ(K̃_(s,a), k̃_(s,a), λ_reg)</code></li>
<li><code>G_(s,a) ← G(Γ_(s,a), Z^π, γ, r, ν, ℓ)</code></li>
<li><code>H_(s,a) ← H(Γ_(s,a), Z^π, γ, r, ν, ℓ)</code></li>
<li><code>Φ_(s,a) ← Φ(K̃_(s′,a′), Γ_(s,a), α̂)</code></li>
</ul>

<li><b>Optimization Step:</b></li>
<ul>
<li>
<code>B_opt ← Optimize(B_init, k̃_(s,a), G_(s,a), H_(s,a), Φ_(s,a), λ_FP, λ_mass)</code>
</li>
</ul>

<li><b>Return:</b> <code>B_opt</code></li>
</ol>

<hr>
