important things that make this work
 - one agent, activated with squashed gaussian (reparameterization trick into tanh)
 - logstd output
 - critic is Q(s, a) and not V(s)
 - EMA of target critic weights instead of hard reset to decrease variance
 - RPB

 Basic stability stuff
 - Gradnorm Clipping
 - Add epsilon

HUMANOID
My SAC @ 10M steps: 8377, v5
Haarnoja 2018 SAC @ 10M steps: ~6500, v1
~29% performance increase
Improvements over original paper:
 - Smooth ELU instead of ReLU
 - Normalized action scale
 - Clamped mean & logstd of policy
 - 