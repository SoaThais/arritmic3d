arritmic3d ./Tests/Refactor_Cases/01_isotropic --slab --nnodes 50 50 5 --spacing 0.2 0.2 0.2 --region-by-side south 1

arritmic3d ./Tests/Refactor_Cases/02_anisotropic --slab --nnodes 50 50 5 --spacing 0.2 0.2 0.2 --region-by-side south 1 --field fibers_orientation '[1,0,0]'

arritmic3d ./Tests/Refactor_Cases/03_S1S2 --slab --nnodes 250 250 5 --spacing 0.2 0.2 0.2 --region-by-side south 1 --field fibers_orientation '[1,0,0]' --region '{"shape":"square","cx":0.0,"cy":0.0,"r1":25.0,"r2":25.0,"activation_region":2}'