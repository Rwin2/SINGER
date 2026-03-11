"""
Entry point for ssv_muilti3dgs_campaign commands.
Delegates to ssv_muilti3dgs_campaign_coruscant (which sets ACADOS env vars).
Run from: /data/erwinpi/SINGER/notebooks/
"""
from ssv_muilti3dgs_campaign_coruscant import app

if __name__ == "__main__":
    app()
