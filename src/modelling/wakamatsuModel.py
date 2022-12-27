from builtins import super
import numpy as np
import numbers
from warnings import warn


class WakamatsuModel(object):
    """Implementation of the differential geometry DLO model by H. Wakamatsu and S. Hirai from the paper:
    Wakamatsu H, Hirai S. Static Modeling of Linear Object Deformation Based on Differential Geometry. The International Journal of Robotics Research. 2004;23(3):293-311


    Attributes
    ----------
    N: int
        Number of ansatz functions. Defaults to N = 10 as Wakamatsu et al. suggest in their paper.

    aPhi: numpy array
        Nx1 array of initial values for the weights to approximate angle Phi. Provide as input to initialize weights. Otherwise defaults to all zero.

    aTheta: numpy array
        Nx1 array of initial values for the weights to approximate angle Theta. Provide as input to initialize weights. Otherwise defaults to all zero.

    aPsi: numpy array
        Nx1 array of initial values for the weights to approximate angle Psi. Provide as input to initialize weights. Otherwise defaults to all zero.

    Rflex: float
        Parameter describing the flexural rigidity of the DLO. Defaults to 1 N/rad.

    Rtor: float
        Parameter describing the trosional rigidity of the DLO. Defaults to 1 N/rad.

    Roh: float
        Parameter describing the specific density (per length) of the DLO. Defaults to 0.1 kg/m.
    """

    def __init__(
        self,
        L=None,
        N=None,
        aPhi=None,
        aTheta=None,
        aPsi=None,
        Rflex=None,
        Rtor=None,
        Roh=None,
        x0=None,
        gravity=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if L is not None and (not isinstance(L, numbers.Number) or L < 0):
            raise ValueError(
                "Expected a positive float for length of the DLO instead got: {}".format(
                    L
                )
            )

        if N is not None and (not isinstance(N, numbers.Number) or N < 1):
            raise ValueError(
                "Expected a positive integer of at least 1 for N instead got: {}".format(
                    N
                )
            )
        elif isinstance(N, numbers.Number) and not isinstance(N, int):
            warn(
                "Received a non-integer value for number of ansatz functions: {}. Casting to integer.".format(
                    N
                )
            )
            N = int(N)

        if aPhi is not None and (not isinstance(aPhi, np.ndarray) or aPhi.size != N):
            raise ValueError(
                "Expected aPhi to be a numpy array of {} weights. Instead got: {} with {} weithgs".format(
                    N, aPhi, aPhi.size
                )
            )

        if aTheta is not None and (
            not isinstance(aTheta, np.ndarray) or aTheta.size != N
        ):
            raise ValueError(
                "Expected aPhi to be a numpy array of {} weights. Instead got: {} with {} weithgs".format(
                    N, aTheta, aTheta.size
                )
            )

        if aPsi is not None and (not isinstance(aPsi, np.ndarray) or aPsi.size != N):
            raise ValueError(
                "Expected aPhi to be a numpy array of {} weights. Instead got: {} with {} weithgs".format(
                    N, aPsi, aPsi.size
                )
            )

        if Rflex is not None and (not isinstance(Rflex, numbers.Number) or Rflex < 0):
            raise ValueError(
                "Expected a positive float for flexibility parameter Rflex instead got: {}".format(
                    Rflex
                )
            )

        if Rtor is not None and (not isinstance(Rtor, numbers.Number) or Rtor < 0):
            raise ValueError(
                "Expected a positive float for torsinal rigidity parameter Rtor instead got: {}".format(
                    Rtor
                )
            )

        if Roh is not None and (not isinstance(Roh, numbers.Number) or Roh < 0):
            raise ValueError(
                "Expected a positive float for density parameter Roh instead got: {}".format(
                    Roh
                )
            )

        if x0 is not None and (not isinstance(x0, np.ndarray) or x0.size != 3):
            raise ValueError(
                "Expected x0 to be a numpy array of x,y,z values with size 3. Instead got: {} with {} weithgs".format(
                    x0, x0.size
                )
            )

        self.L = 1 if L is None else L
        self.N = 10 if N is None else N
        self.aPhi = 0 * np.ones(self.N) if aPhi is None else aPhi
        self.aTheta = 0 * np.ones(self.N) if aTheta is None else aTheta
        self.aPsi = 0 * np.ones(self.N) if aPsi is None else aPsi
        self.Rflex = 1 if Rflex is None else Rflex
        self.Rtor = 1 if Rtor is None else Rtor
        self.Roh = 0.1 if Roh is None else Roh
        self.x0 = np.zeros(3) if x0 is None else x0
        self.gravity = np.array([0, 0, 9.81]) if gravity is None else gravity

    def evalAnsatzFuns(self, S):
        """returns the ansatz functions evaluated at the local coodinates in S

        Args:
            S (np.array): Array of local coordinates in [0,L] where the ansatz functions should be evaluated

        Returns:
            E (np.array): NxD array of ansatz functions evaluated at local coodinates in S
        """
        E = np.ones((self.N, len(S)))
        E[1, :] = S / self.L
        for i in range(1, int((self.N / 2))):
            E[2 * i, :] = np.sin(2 * np.pi * i * S / self.L)
            E[2 * i + 1, :] = np.cos(2 * np.pi * i * S / self.L)
        return E

    def evalAnsatzFunDerivs(self, S):
        """returns the derivatives of the ansatz functions evaluated at the local coodinates in S
        Args:
            S (np.array): Array of local coordinates in [0,L] where the ansatz functions should be evaluated
        Returns:
            dE (np.array): (S.size)xD array of derivatives of the ansatz functions evaluated at local coodinates in S
        """
        dE = np.zeros((self.N, len(S)))
        dE[1, :] = np.ones(len(S)) / self.L
        for i in range(1, int((self.N / 2))):
            dE[2 * i, :] = np.cos(2 * np.pi * i * S / self.L) * (2 * np.pi * i / self.L)
            dE[2 * i + 1, :] = -np.sin(2 * np.pi * i * S / self.L) * (
                2 * np.pi * i / self.L
            )
        return dE

    def evalTheta(self, S):
        """returns the functional of anlge Theta evaluated at the local coodinates in S
        Args:
            S (np.array): Array of local coordinates in [0,L]
        Returns:
            theta (np.array): (S.size)x1 array of angle values corresponding to the local coodinates in S
        """
        theta = self.aTheta @ self.evalAnsatzFuns(S)
        return theta

    def evalThetaDervi_S(self, S):
        """returns the derivative of the functional of anlge Theta with resprect to the local coordinates (dTheta/ds) evaluated at the local coodinates in S
        Args:
            S (np.array): Array of local coordinates in [0,L]
        Returns:
            dTheta (np.array): (S.size)x1 array of angle changes corresponding to the local coodinates in S
        """
        return self.aTheta @ self.evalAnsatzFunDerivs(S)

    def evalPhi(self, S):
        """returns the functional of anlge Phi evaluated at the local coodinates in S
        Args:
            S (np.array): Array of local coordinates in [0,L]
        Returns:
            phi (np.array): (S.size)x1 array of angle values corresponding to the local coodinates in S
        """
        phi = self.aPhi @ self.evalAnsatzFuns(S)
        return phi

    def evalPhiDeriv_S(self, S):
        """returns the derivative of the functional of anlge Phi with resprect to the local coordinates (dPhi/ds) evaluated at the local coodinates in S
        Args:
            S (np.array): Array of local coordinates in [0,L]
        Returns:
            dPhi (np.array): (S.size)x1 array of angle changes corresponding to the local coodinates in S
        """
        return self.aPhi @ self.evalAnsatzFunDerivs(S)

    def evalPsi(self, S):
        """returns the functional of anlge Psi evaluated at the local coodinates in S
        Args:
            S (np.array): Array of local coordinates in [0,L]
        Returns:
            psi (np.array): (S.size)x1 array of angle values corresponding to the local coodinates in S
        """
        psi = self.aPsi @ self.evalAnsatzFuns(S)
        return psi

    def evalPsiDeriv_S(self, S):
        """returns the derivative of the functional of anlge Psi with resprect to the local coordinates (dPsi/ds) evaluated at the local coodinates in S
        Args:
            S (np.array): Array of local coordinates in [0,L]
        Returns:
            dPsi (np.array): (S.size)x1 array of angle changes corresponding to the local coodinates in S
        """
        return self.aPsi @ self.evalAnsatzFunDerivs(S)

    def evalZeta(self, S):
        """returns the unit tangent vector zeta of the local coordinate system evaluated at the local coodinates in S.
        Note: we assume zero strain (see eq.5)
        Args:
            S (np.array): Array of local coordinates in [0,L]
        Returns:
            zeta (np.array): (S.size)x3 array of vectors corresponding to the local coodinates in S
        """
        return np.array(
            (
                np.sin(self.evalTheta(S)) * np.cos(self.evalPhi(S)),
                np.sin(self.evalTheta(S)) * np.sin(self.evalPhi(S)),
                np.cos(self.evalTheta(S)),
            )
        )

    def evalZetaDeriv_aTheta(self, S):
        """returns the derivative of the unit tangent vector zeta with resprect to the changes of weights aTheta (dZeta/daTheta) evaluated at the local coodinates in S.
        Args:
            S (np.array): Array of local coordinates in [0,L]
        Returns:
            dZeta_daTheta (np.array): (S.size)x3 array of derivatives corresponding to the local coodinates in S
        """
        dZeta_daTheta = np.zeros([3, (S.size)])
        dZeta_daTheta[0, :] = np.cos(self.evalTheta(S)) * np.cos(self.evalPhi(S))
        dZeta_daTheta[1, :] = np.cos(self.evalTheta(S)) * np.sin(self.evalPhi(S))
        dZeta_daTheta[2, :] = -np.sin(self.evalTheta(S))
        return dZeta_daTheta

    def evalZetaDeriv_aPhi(self, S):
        """returns the derivative of the unit tangent vector zeta with resprect to the changes of weights aPhi (dZeta/daPhi) evaluated at the local coodinates in S.
        Args:
            S (np.array): Array of local coordinates in [0,L]
        Returns:
            dZeta_daPhi (np.array): (S.size)x3 array of derivatives corresponding to the local coodinates in S
        """
        dZeta_daPhi = np.zeros([3, (S.size)])
        dZeta_daPhi[0, :] = -np.sin(self.evalTheta(S)) * np.sin(self.evalPhi(S))
        dZeta_daPhi[1, :] = np.sin(self.evalTheta(S)) * np.cos(self.evalPhi(S))
        dZeta_daPhi[2, :] = 0
        return dZeta_daPhi

    def evalKappaSquared(self, S):
        """returns the squared curvature kappa^2 evaluated at the local coodinates in S.
        Args:
            S (np.array): Array of local coordinates in [0,L]
        Returns:
            kappaSquared (np.array): (S.size)x1 array of squared curvature values corresponding to the local coodinates in S
        """
        dThetaSquared = np.square(self.aTheta @ self.evalAnsatzFunDerivs(S))
        dPhiSquared = np.square(self.aPhi @ self.evalAnsatzFunDerivs(S))
        return dThetaSquared + dPhiSquared * np.square(np.sin(self.evalTheta(S)))

    def evalKappaSquaredDeriv_aTheta(self, S):
        """returns the jacobian of the squared curvature with resprect to parameters aTheta (dkappa^2/daTheta) evaluated at the local coodinates in S.
        Args:
            S (np.array): Array of local coordinates in [0,L]
        Returns:
            dkappaSquared_daTheta (np.array): (S.size)xN array of squared curvature jacobians corresponding to the local coodinates in S
        """
        return 2 * (
            self.aTheta * np.square(self.evalAnsatzFunDerivs(S)).T
        ).T + np.square(self.aPhi @ self.evalAnsatzFunDerivs(S)) * 2 * np.sin(
            self.evalTheta(S)
        ) * np.cos(
            self.evalTheta(S)
        ) * self.evalAnsatzFuns(
            S
        )

    def evalKappaSquaredDeriv_aPhi(self, S):
        """returns the jacobian of the squared curvature with resprect to parameters aPhi(dkappa^2/daPhi) evaluated at the local coodinates in S.
        Args:
            S (np.array): Array of local coordinates in [0,L]
        Returns:
            dkappaSquared_daPhi (np.array): (S.size)xN array of squared curvature jacobians corresponding to the local coodinates in S
        """
        return (
            2
            * self.aPhi
            * (
                np.square(self.evalAnsatzFunDerivs(S))
                * np.square(np.sin(self.evalTheta(S)))
            ).T
        ).T

    def evalOmegaSquared(self, S):
        """returns the squared torsion omega^2 evaluated at the local coodinates in S.
        Args:
            S (np.array): Array of local coordinates in [0,L]
        Returns:
            omegaSquared (np.array): (S.size)x1 array of squared torsion values corresponding to the local coodinates in S
        """
        dPhi = self.aPhi @ self.evalAnsatzFunDerivs(S)
        dPsi = self.aPsi @ self.evalAnsatzFunDerivs(S)
        omegaSquared = np.square(dPhi * np.cos(self.evalTheta(S)) + dPsi)
        # print(omegaSquared)
        return omegaSquared

    def evalOmegaSquaredDeriv_aTheta(self, S):
        """returns the jacobian of the squared torsion with resprect to parameters aTheta (domega^2/daTheta) evaluated at the local coodinates in S.
        Args:
            S (np.array): Array of local coordinates in [0,L]
        Returns:
            domegaSquared_daTheta (np.array): (S.size)xN array of squared torsion jacobians corresponding to the local coodinates in S
        """
        return (
            (
                2 * (self.evalPhiDeriv_S(S)) * np.cos(self.evalTheta(S))
                + self.evalPsiDeriv_S(S)
            )
            * self.evalPhiDeriv_S(S)
            * (-np.sin(self.evalTheta(S)))
            * self.evalAnsatzFuns(S)
        )

    def evalOmegaSquaredDeriv_aPhi(self, S):
        """returns the jacobian of the squared torsion with resprect to parameters aPhi (domega^2/daPhi) evaluated at the local coodinates in S.
        Args:
            S (np.array): Array of local coordinates in [0,L]
        Returns:
            domegaSquared_daPhi (np.array): (S.size)xN array of squared torsion jacobians corresponding to the local coodinates in S
        """
        return (
            2
            * (
                self.evalPhiDeriv_S(S) * np.cos(self.evalTheta(S))
                + self.evalPsiDeriv_S(S)
            )
            * self.evalAnsatzFunDerivs(S)
            * np.cos(self.evalTheta(S))
        )

    def evalOmegaSquaredDeriv_aPsi(self, S):
        """returns the jacobian of the squared torsion with resprect to parameters aPsi (domega^2/daPsi) evaluated at the local coodinates in S.
        Args:
            S (np.array): Array of local coordinates in [0,L]
        Returns:
            domegaSquared_daPsi (np.array): (S.size)xN array of squared torsion jacobians corresponding to the local coodinates in S
        """
        return (
            2
            * (
                self.evalPhiDeriv_S(S) * np.cos(self.evalTheta(S))
                + self.evalPsiDeriv_S(S)
            )
            * self.evalAnsatzFunDerivs(S)
        )

    def evalPosition(self, s, numEvaluationPoints):
        """returns the position of a point on the DLO evaluated at the local coodinate s.
        Args:
            s (float): local coordinate in [0,L]
            numEvaluationPoints (int): number of evaluation points
        Returns:
            x (np.array): 1x3 array of positions corresponding to the local coodinate s
        """
        sEval = np.linspace(0, s, numEvaluationPoints)
        x = self.x0 + np.trapz(self.evalZeta(sEval), sEval)
        return x

    def evalPositions(self, S, numEvaluationPoints):
        """returns the positions of points on the DLO evaluated at the local coodinates S.
        Args:
            S (np.array): local coordinates in [0,L]
            numEvaluationPoints (list): list of numer of evaluation points
        Returns:
            X (np.array): (S.size)x3 array of positions corresponding to the local coodinates in S
        """
        X = np.zeros((S.size, 3))
        for i, s in enumerate(S):
            X[i, :] = self.evalPosition(s, numEvaluationPoints[i])
        return X

    def evalUflex(self, s, numEvaluationPoints):
        """returns the flexural energy of the DLO integrated up to a local coodinate s.
        Args:
            s (float): local coordinate in [0,L]
        Returns:
            Uflex (flaot): flexural energy accumulated up to the local coodinate s
        """
        sEval = np.linspace(0, s, numEvaluationPoints)
        Uflex = 0.5 * self.Rflex * np.trapz(self.evalKappaSquared(sEval), sEval)
        return Uflex

    def evalUtor(self, s, numEvaluationPoints):
        """returns the torsional energy of the DLO integrated up to a local coodinate s.
        Args:
            s (float): local coordinate in [0,L]
        Returns:
            Uflex (flaot): torsional energy accumulated up to the local coodinate s
        """
        sEval = np.linspace(0, s, numEvaluationPoints)
        Utor = 0.5 * self.Rtor * np.trapz(self.evalOmegaSquared(sEval), sEval)
        return Utor

    def evalUgrav(self, s, numEvaluationPoints):
        """returns the gravitational energy of the DLO integrated up to a local coodinate s.
        Args:
            s (float): local coordinate in [0,L]
        Returns:
            Ugrav (flaot): gravitational energy accumulated up to the local coodinate s
        """
        sEval = np.linspace(0, s, numEvaluationPoints)
        numIntegrationPointList = [2, 2] + [
            *range(2, sEval.size)
        ]  # make sure the first positions have at least two integration points for numerical integration
        XGravity = self.evalPositions(sEval, numIntegrationPointList)
        Ugrav = self.Roh * np.trapz(XGravity @ self.gravity, sEval)
        return Ugrav
