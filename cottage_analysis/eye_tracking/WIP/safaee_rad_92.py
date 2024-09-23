import numpy as np
from numpy.polynomial import Polynomial
from cottage_analysis.eye_tracking import eye_model_fitting


def conic(ellipse):
    # NOT IN USE. ATTEMPT TO FOLLOW Safaee-Rad 1992
    ax = np.cos(ellipse.angle)
    ay = np.sin(ellipse.angle)

    a2 = ellipse.major_radius**2
    b2 = ellipse.minor_radius**2

    A = ax * ax / a2 + ay * ay / b2
    B = 2 * ax * ay / a2 - 2 * ax * ay / b2
    C = ay * ay / a2 + ax * ax / b2
    D = (-2 * ax * ay * ellipse.centre_y - 2 * ax * ax * ellipse.centre_x) / a2 + (
        2 * ax * ay * ellipse.centre_y - 2 * ay * ay * ellipse.centre_x
    ) / b2
    E = (-2 * ax * ay * ellipse.centre_x - 2 * ay * ay * ellipse.centre_y) / a2 + (
        2 * ax * ay * ellipse.centre_x - 2 * ax * ax * ellipse.centre_y
    ) / b2
    F = (
        (
            2 * ax * ay * ellipse.centre_x * ellipse.centre_y
            + ax * ax * ellipse.centre_x * ellipse.centre_x
            + ay * ay * ellipse.centre_y * ellipse.centre_y
        )
        / a2
        + (
            -2 * ax * ay * ellipse.centre_x * ellipse.centre_y
            + ay * ay * ellipse.centre_x * ellipse.centre_x
            + ax * ax * ellipse.centre_y * ellipse.centre_y
        )
        / b2
        - 1
    )
    return pd.Series(dict(A=A, B=B, C=C, D=D, E=E, F=F))


def conicoid(conic, vertex):
    # NOT IN USE. ATTEMPT TO FOLLOW Safaee-Rad 1992
    alpha, beta, gamma = vertex
    A = gamma**2 * conic.A
    B = gamma**2 * conic.C
    C = (
        conic.A * alpha**2
        + conic.B * alpha * beta
        + conic.C * beta**2
        + conic.D * alpha
        + conic.E * beta
        + conic.F
    )
    F = -gamma * (conic.C * beta + conic.B / 2 * alpha + conic.E / 2)
    G = -gamma * (conic.B / 2 * beta + conic.A * alpha + conic.D / 2)
    H = gamma**2 * conic.B / 2
    U = gamma**2 * conic.D / 2
    V = gamma**2 * conic.E / 2
    W = -gamma * (conic.E / 2 * beta + conic.D / 2 * alpha + conic.F)
    D = gamma**2 * conic.F
    return A, B, C, D, F, G, H, U, V, W


def unproject(ellipse, circle_radius, focal_length):
    # NOT IN USE. ATTEMPT TO FOLLOW Safaee-Rad 1992
    cam_centre_in_ellipse = np.array([0, 0, -focal_length])
    cone_base = conic(ellipse)
    pupil_cone = conicoid(cone_base, cam_centre_in_ellipse)
    a, b, c, d, f, g, h, u, v, w = pupil_cone

    # Get canonical conic form:
    #     lambda(1) X^2 + lambda(2) Y^2 + lambda(3) Z^2 = mu
    # Safaee-Rad 1992 eq (6)
    # Done by solving the discriminating cubic (10)
    # Lambdas are sorted descending because order of roots doesn't
    # matter, and it later eliminates the case of eq (30), where
    # lambda(2) > lambda(1)

    cubic = Polynomial(
        [
            -(a * b * c + 2 * f * g * h - a * f**2 - b * g**2 - c * h**2),
            (b * c + c * a + a * b - f**2 - g**2 - h**2),
            -(a + b + c),
            1.0,
        ],
    )
    lambdas = np.sort(cubic.roots())[::-1]

    assert lambdas[0] >= lambdas[1]
    assert lambdas[1] > 0
    assert lambdas[2] < 0

    # Now want to calculate l,m,n of the plane
    #     lX + mY + nZ = p
    # which intersects the cone to create a circle.
    # Safaee-Rad 1992 eq (31)
    # [Safaee-Rad 1992 eq (33) comes out of this as a result of lambdas[1] == lambdas[2]]
    n = np.sqrt((lambdas[1] - lambdas[2]) / (lambdas[0] - lambdas[2]))
    m = 0.0
    l = np.sqrt((lambdas[0] - lambdas[1]) / (lambdas[0] - lambdas[2]))

    # Want to calculate T1, the rotation transformation from image
    # space in the canonical conic frame back to image space in the
    # real world

    # Safaee-Rad 1992 eq (12)
    t1 = (b - lambdas) * g - f * h
    t2 = (a - lambdas) * f - g * h
    t3 = -(a - lambdas) * (t1 / t2) / g - h / g

    mi = 1 / np.sqrt(1 + (t1 / t2) ** 2 + t3**2)
    li = (t1 / t2) * mi
    ni = t3 * mi

    # Safaee-Rad 1992 eq (8)
    T1 = np.identity(4)
    T1[:3, :3] = np.vstack([li, mi, ni])

    # If li,mi,ni follow the left hand rule, flip their signs
    if np.dot(np.cross(li, mi), ni) < 0:
        li = -li
        mi = -mi
        ni = -ni

    # Calculate T2, a translation transformation from the canonical
    # conic frame to the image space in the canonical conic frame
    # Safaee-Rad 1992 eq (14)
    T2 = np.identity(4)
    T2[:3, 3] = -(u * li + v * mi + w * ni) / lambdas

    ls = [l, -l]
    solutions = []
    for i, l in enumerate(ls):
        # Circle normal in image space (i.e. gaze vector)
        gaze = T1 @ np.array([l, m, n, 1])

        # Calculate T3, a rotation from a frame where Z is the circle normal
        # to the canonical conic frame
        # Safaee-Rad 1992 eq (19)
        # Want T3 = / -m/sqrt(l*l+m*m) -l*n/sqrt(l*l+m*m) l \
        #              |  l/sqrt(l*l+m*m) -m*n/sqrt(l*l+m*m) m |
        #                \            0           sqrt(l*l+m*m)   n /
        # But m = 0, so this simplifies to
        #      T3 = /       0      -n*l/sqrt(l*l) l \
        #           |  l/sqrt(l*l)        0       0 |
        #           \          0         sqrt(l*l)   n /
        #         = /    0    -n*sgn(l) l \
        #           |  sgn(l)     0     0 |
        #           \       0       |l|    n /

        if l == 0:
            # Discontinuity of sgn(l), have to handle explicitly
            assert n == 1
            print("Warning: l == 0")
            T3 = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        else:
            sgnl = np.sign(l)
            T3 = np.array(
                [
                    [0, -n * np.sign(l), l, 0],
                    [np.sign(l), 0, 0, 0],
                    [0, abs(l), n, 0],
                    [0, 0, 0, 1],
                ]
            )

        # Calculate the circle centre
        # Safaee-Rad 1992 eq (38), using T3 as defined in (36)
        A = lambdas @ T3[:3, 0] ** 2
        B = lambdas @ (T3[:3, 0] * T3[:3, 2])
        C = lambdas @ (T3[:3, 1] * T3[:3, 2])
        D = lambdas @ T3[:3, 2] ** 2

        # Safaee-Rad 1992 eq (41)
        Z0 = A * circle_radius / np.sqrt(B**2 + C**2 - A * D)
        X0 = -B / A * Z0
        Y0 = -C / A * Z0
        centre_in_Xprime = np.array([X0, Y0, Z0, 1])

        # Safaee-Rad 1992 eq (34)
        T0 = np.identity(4)
        T0[2, 3] = focal_length

        # Safaee-Rad 1992 eq (42) using (35)
        centre = T0 @ T1 @ T2 @ T3 @ centre_in_Xprime

        # If z is negative (behind the camera), choose the other
        # solution of eq (41) [maybe there's a way of calculating which
        # solution should be chosen first]

        if centre[2] < 0:
            centre_in_Xprime = -centre_in_Xprime
            centre = T0 @ T1 @ T2 @ T3 @ centre_in_Xprime

        # Make sure that the gaze vector is toward the camera and is normalised
        if np.dot(gaze, centre) > 0:
            gaze[:3] = -gaze[:3]

        gaze /= np.linalg.norm(gaze[:3])

        # Save the results
        solutions.append(dict(centre=centre, gaze=gaze, circle_radius=circle_radius))
    return solutions


def project_point(point, focal_length):
    # NOT IN USE. ATTEMPT TO FOLLOW Safaee-Rad 1992
    return focal_length * point[:2] / point[2]


def unproject_observations(ellipses, circle_radius, focal_length):
    # NOT IN USE. ATTEMPT TO FOLLOW Safaee-Rad 1992
    """Unproject

    Based on SingleEyeFitter.cpp, unproject_observations, L1595

    Args:
        ellipse (_type_): _description_
        circle_radius (_type_): _description_
        focal_length (_type_): _description_
    """
    pupil_unprojection_pairs = []
    pupil_gazelines_proj = []
    for ellipse in ellipses:
        unprojection_pair = unproject(
            ellipse=ellipse, circle_radius=circle_radius, focal_length=focal_length
        )

        # Get projected circles and gaze vectors
        #
        # Project the circle centres and gaze vectors down back onto the image
        # plane. We're only using them as line parametrisations, so it doesn't
        # matter which of the two centres/gaze vectors we use, as the
        # two gazes are parallel and the centres are co-linear.

        c = unprojection_pair[0]["centre"]
        v = unprojection_pair[0]["gaze"]

        c_proj = project_point(c, focal_length)
        v_proj = project_point(v + c, focal_length) - c_proj

        v_proj /= np.linalg.norm(v_proj)

        pupil_unprojection_pairs.append(unprojection_pair)
        pupil_gazelines_proj.append([c_proj, v_proj])

    # Get eyeball centre
    #
    # Find a least-squares 'intersection' (point nearest to all lines) of
    # the projected 2D gaze vectors. Then, unproject that circle onto a
    # point a fixed distance away.
    #
    # For robustness, use RANSAC to eliminate stray gaze lines
    #
    # (This has to be done here because it's used by the pupil circle
    # disambiguation)


if __name__ == "__main__":
    import os
    from pathlib import Path
    import pandas as pd
    import flexiznam as flz

    raw = Path(flz.PARAMETERS["data_root"]["raw"])
    processed = Path(flz.PARAMETERS["data_root"]["processed"])
    project = "hey2_3d-vision_foodres_20220101"
    mouse = "PZAH6.4b"
    session = "S20220419"
    recording = "R145152_SpheresPermTubeReward"
    camera = "right_eye_camera"

    data_path = processed / project / mouse / session / recording / camera
    dlc_path = data_path  # / "dlc_output"
    ellipse_fits = None
    print("Fitting ellipses")
    for fname in dlc_path.glob("*.h5"):
        if "filtered" in fname.name:
            continue
        if ellipse_fits is not None:
            raise IOError("Multiple DLC results files")

        fit_save = dlc_path / "{0}_ellipse_fits.csv".format(fname.stem)
        if fit_save.exists():
            ellipse_fits = pd.read_csv(fit_save)
        else:
            ellipse_fits = eye_model_fitting.fit_ellipses(fname)
            ellipse_fits.to_csv(fit_save, index=False)
    raise NotImplementedError
    # let's unproject
    print("Unprojecting")
    px_per_mm = 5
    focal_length = 1 / (10.0 * px_per_mm)
    ellipses = [ellipse for _, ellipse in ellipse_fits.iterrows()]
    all_sol = []
    for i_el, ellipse in ellipse_fits.iterrows():
        solutions = unproject(ellipse, circle_radius=30, focal_length=focal_length)

        # ok we did the unprojection
        print(1 + 1)
        all_sol.append(solutions)
