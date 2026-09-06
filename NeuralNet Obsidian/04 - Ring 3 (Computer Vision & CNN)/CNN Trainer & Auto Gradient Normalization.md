# Plain English Summary

Training deep convolutional networks can suffer from two major problems: gradients can explode (becoming huge and causing the model to diverge) or vanish (becoming so tiny that the model never learns). Furthermore, models can repeatedly make the same mistake in training cycles.

The `CNNTrainer` solves these challenges by combining four advanced technologies:
1. **Automatic Gradient Normalization**: Measures the total energy (magnitude) of the gradients across all layers and standardizes it to a healthy target scale, preventing both exploding and vanishing gradients.
2. **Taylor Loss-Trajectory Foresight**: Looks at the curvature of recent loss values to predict upcoming rebounds or overshoots and modulates the learning rate ahead of time.
3. **Online Meta-LR Optimization**: Uses a meta-neural controller to adjust the step size based on real-time loss change dynamics.
4. **Episodic Mistake Memory with Repulsive Barriers & Dynamic Sizing**: Remembers gradient directions from past mistake spikes, pushes upcoming gradients away from those failure states, and dynamically expands network layer capacity if the model repeatedly stumbles.

---

# CNN Trainer & Auto Gradient Normalization

## 1. Automatic Gradient Normalization

Let $G \in \mathbb{R}^P$ be the concatenated global gradient vector of all $P$ learnable parameters (both Conv2D filters/biases and Dense weights/biases):

$$G = \left[ \nabla_{W_{\text{conv}}}, \nabla_{b_{\text{conv}}}, \nabla_{W_{\text{dense}}}, \nabla_{b_{\text{dense}}} \right]^T$$

### Global Norm & Unit Vector Extraction
$$\|G\|_2 = \sqrt{\sum_{i=1}^P G_i^2}, \quad \hat{U} = \frac{G}{\|G\|_2 + \epsilon}$$

### Adaptive Normalization Scaling
Given target energy $G_{\text{target}}$ and ceiling $G_{\text{ceiling}}$:

$$s = \frac{G_{\text{target}}}{\max(\epsilon, \|G\|_2)}$$

$$\text{if } s \cdot \|G\|_2 > G_{\text{ceiling}}, \quad s = \frac{G_{\text{ceiling}}}{\|G\|_2}$$

$$G_{\text{normalized}} = \hat{U} \cdot (\|G\|_2 \cdot s) = \hat{U} \cdot G_{\text{target}}$$

This guarantees consistent, well-calibrated step sizes across heterogeneous convolutional and fully connected layers.

---

## 2. Episodic Mistake Memory & Repulsive Gradient Barriers

When a batch produces an anomalous loss spike ($\mathcal{L}_{\text{batch}} > \tau \cdot \mathcal{L}_{\text{EMA}}$), the unit gradient direction $\hat{U}_{\text{failure}}$ is archived into the mistake memory buffer:

$$\mathcal{M} = \left\{ (\mathcal{L}_k, \hat{U}_k) \right\}_{k=1}^K$$

### Repulsive Force Computation
During subsequent training steps, the active gradient vector is compared against all stored failure directions via cosine similarity:

$$\rho_k = \hat{U} \cdot \hat{U}_k$$

$$\text{if } \rho_k > 0: \quad G \leftarrow G - \beta_{\text{repel}} \sqrt{\rho_k} \cdot \hat{U}_k$$

This creates an orthogonal repulsive barrier in parameter space, steering the optimizer away from previously discovered failure valleys.

---

## 3. Dynamic Sizing on Mistake Streaks

If the active gradient exhibits high alignment ($\rho > 0.40$) with past mistakes for consecutive epochs ($N_{\text{streak}} \ge N_{\text{trigger}}$), the trainer signals capacity exhaustion and triggers automatic architectural expansion:

$$\text{model.expand\_capacity}(\gamma_{\text{growth}})$$

Where $\gamma_{\text{growth}} = 1.25$ increases both convolutional channel depth and dense hidden layer units by 25%.

---

## 4. Multi-Foresight Applied Learning Rate

The effective step size incorporates both Taylor polynomial curvature foresight and Meta-Neural loss prediction:

$$\eta_{\text{applied}} = \eta_0 \cdot \lambda_{\text{Taylor}}(\mathcal{H}_{\mathcal{L}}) \cdot \mu_{\text{Meta}}(\mathcal{L}, \Delta \mathcal{L}, \|G\|)$$

$$W_{t+1} = W_t - \eta_{\text{applied}} \left( G_{\text{normalized}} + \lambda_{\text{decay}} W_t \right)$$
