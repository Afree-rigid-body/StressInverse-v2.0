

def calculate_fault_stress_from_principal_stresses(sigma_values, sigma_vectors, strike, dip):
    import numpy as np
    """
    M-dM-=M-?M-gM-^TM-(M-dM-8M-;M-eM-:M-^TM-eM-^JM-^[M-gM-^ZM-^DM-eM-$M-'M-eM-0M-^OM-eM-^RM-^LM-fM-^VM-9M-eM-^PM-^QM-fM-^]M-%M-hM-.M-!M-gM-.M-^WM-gM-^IM-9M-eM-.M-^ZM-fM-^VM--M-eM-1M-^BM-iM-^]M-"M-dM-8M-^JM-gM-^ZM-^DM-fM--M-#M-eM-:M-^TM-eM-^JM-^[M-eM-^RM-^LM-eM-^IM-*M-eM-:M-^TM-eM-^JM-^[M-cM-^@M-^B
    
    M-eM-^OM-^BM-fM-^UM-0:
    - sigma_values: M-dM-8M-;M-eM-:M-^TM-eM-^JM-^[M-gM-^ZM-^DM-eM-$M-'M-eM-0M-^OM-oM-<M-^LM-eM-=M-"M-eM-<M-^OM-dM-8M-:[M-OM-^C1, M-OM-^C2, M-OM-^C3]
    - sigma_vectors: M-dM-8M-;M-eM-:M-^TM-eM-^JM-^[M-gM-^ZM-^DM-fM-^VM-9M-eM-^PM-^QM-eM-^PM-^QM-iM-^GM-^OM-oM-<M-^LM-eM-=M-"M-eM-<M-^OM-dM-8M-:3x3M-gM-^ZM-^DM-fM-^UM-0M-gM-;M-^DM-oM-<M-^LM-fM-/M-^OM-dM-8M-^@M-eM-^HM-^WM-fM-^XM-/M-dM-8M-^@M-dM-8M-*M-dM-8M-;M-eM-:M-^TM-eM-^JM-^[M-gM-^ZM-^DM-fM-^VM-9M-eM-^PM-^QM-eM-^PM-^QM-iM-^GM-^O
    - strike: M-fM-^VM--M-eM-1M-^BM-iM-^]M-"M-gM-^ZM-^DM-hM-5M-0M-eM-^PM-^QM-oM-<M-^LM-dM-;M-%M-eM-:M-&M-dM-8M-:M-eM-M-^UM-dM-=M-
    - dip: M-fM-^VM--M-eM-1M-^BM-iM-^]M-"M-gM-^ZM-^DM-eM-^@M->M-hM-'M-^RM-oM-<M-^LM-dM-;M-%M-eM-:M-&M-dM-8M-:M-eM-M-^UM-dM-=M-
    
    M-hM-?M-^TM-eM-^[M-^^:
    - tau_normal: M-fM-^VM--M-eM-1M-^BM-iM-^]M-"M-dM-8M-^JM-gM-^ZM-^DM-fM--M-#M-eM-:M-^TM-eM-^JM-^[
    - tau_shear: M-fM-^VM--M-eM-1M-^BM-iM-^]M-"M-dM-8M-^JM-gM-^ZM-^DM-eM-^IM-*M-eM-:M-^TM-eM-^JM-^[
    """
    # M-fM-^VM--M-eM-1M-^BM-iM-^]M-"M-fM-3M-^UM-eM-^PM-^QM-iM-^GM-^OM-gM-^ZM-^DM-hM-.M-!M-gM-.M-^W
    n = np.array([
        -np.sin(np.radians(dip)) * np.sin(np.radians(strike)),
        np.sin(np.radians(dip)) * np.cos(np.radians(strike)),
        -np.cos(np.radians(dip))
    ])
    
    # M-dM-=M-?M-gM-^TM-(M-gM-^IM-9M-eM->M-^AM-eM-^PM-^QM-iM-^GM-^OM-eM-^RM-^LM-gM-^IM-9M-eM->M-^AM-eM-^@M-<M-iM-^GM-M-fM-^-^DM-eM-:M-^TM-eM-^JM-^[M-eM-<M- M-iM-^GM-^O
    stress_tensor = np.dot(sigma_vectors, np.dot(np.diag(sigma_values), sigma_vectors.T))
    
    # M-hM-.M-!M-gM-.M-^WM-fM-^VM--M-eM-1M-^BM-iM-^]M-"M-dM-8M-^JM-gM-^ZM-^DM-fM--M-#M-eM-:M-^TM-eM-^JM-^[
    tau_normal = np.dot(n, np.dot(stress_tensor, n))
    
    # M-hM-.M-!M-gM-.M-^WM-fM-^VM--M-eM-1M-^BM-iM-^]M-"M-dM-8M-^JM-gM-^ZM-^DM-eM-^IM-*M-eM-:M-^TM-eM-^JM-^[
    tau_total = np.dot(stress_tensor, n)
    tau_shear_magnitude = np.sqrt(np.dot(tau_total, tau_total) - tau_normal**2)
    
    return tau_normal, tau_shear_magnitude



