important things that make this work
 - one agent, activated with squashed gaussian (reparameterization trick into tanh)
 - logstd output
 - critic is Q(s, a) and not V(s)
 - EMA of target critic weights instead of hard reset to decrease variance
 - RPB

 Basic stability stuff
 - Gradnorm Clipping
 - Add epsilon