#!/usr/bin/env python3
"""Version and import check executed as file to avoid PowerShell -c quoting bugs."""
from __future__ import annotations
import sys

def main():
    print('V23_PYTHON_EXE=' + sys.executable, flush=True)
    print('V23_PYTHON_VERSION=' + sys.version.split()[0], flush=True)
    try:
        import numpy as np
        import scipy
        from scipy.integrate import solve_ivp
    except (ImportError, OSError) as e:
        print('V23_DEPENDENCY_IMPORT_FAILED=' + repr(e), file=sys.stderr, flush=True)
        return 12
    # Execute a real, tiny SciPy BDF integration to verify the solver's binary
    # compatibility rather than checking only import names or pip metadata.
    import numpy as np
    sol = solve_ivp(lambda t, y: -y, (0.0, 0.5), np.array([1.0]), method='BDF',
                    rtol=1e-7, atol=1e-9)
    if not sol.success or not np.isfinite(sol.y).all() or abs(float(sol.y[0,-1]) - float(np.exp(-0.5))) > 1e-5:
        print('V23_DEPENDENCY_BDF_SMOKE_FAILED=' + str(sol.message), file=sys.stderr, flush=True)
        return 13
    print('V23_DEPENDENCY_PASS numpy=' + np.__version__ + ' scipy=' + scipy.__version__ +
          ' BDF_smoke=PASS', flush=True)
    return 0

if __name__ == '__main__':
    sys.exit(main())
