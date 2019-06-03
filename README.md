Experimental implementation for "generic" rollback implementation for TS.

This allows TSRollBack() for implementations that don't support it for free,
by composing an extra vector with the TS object, and using a pre-step hook.

Here we use this to roll back and stop based on a simple criterion in
a  post-step hook.

Try
  
    make
    ./runme -ts_monitor -ts_view -ts_type sundials 
