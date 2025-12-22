import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import numpy.linalg as lin

# from scipy.integrate import odeint


def zad1_1_1():
    C1 = 1
    C2 = 1 / 2
    R1 = 2
    R2 = 4

    A = np.array([[-1 / (R1 * C1), 0], [0, -1 / (R2 * C2)]])
    B = np.array([[1 / (R1 * C1)], [1 / (R2 * C2)]])
    C = [1, 1]
    D = 0

    K = np.hstack([B, A @ B])
    wiersze = K.shape[0]

    if lin.matrix_rank(K) == wiersze:
        print("Układ oryginalny jest STEROWALNY")
        print(f"rank(K)={lin.matrix_rank(K)}")
        print(f"n={wiersze}")
    else:
        print("Układ oryginalny jest NIESTEROWALNY")
        print(f"rank(K)={lin.matrix_rank(K)}")
        print(f"n={wiersze}")

    system = sp.StateSpace(A, B, C, D)

    # P_1 = np.hstack([B, A @ B])

    # A1 = lin.inv(P_1) @ A @ P_1
    # B1 = lin.inv(P_1) @ B
    # C1_reg = np.array(C) @ P_1
    # D1 = D

    # sys = sp.StateSpace(A1, B1, C1_reg, D1)

    # K_reg = np.hstack([B1, A1 @ B1])
    # wiersze_reg = K_reg.shape[0]

    # if lin.matrix_rank(K_reg) == wiersze_reg:
    #     print("Układ w postaci regulatorowej jest STEROWALNY")
    # else:
    #     print("Układ w postaci regulatorowej jest NIESTEROWALNY")

    t = np.linspace(0, 100, 1000)

    t, y = sp.step(system, T=t)

    u2 = np.ones_like(t) * 2
    t2, y2, _ = sp.lsim(system, U=u2, T=t)

    u3 = np.sin(t) - 0.5
    t3, y3, _ = sp.lsim(system, U=u3, T=t)

    plt.figure(figsize=(10, 10))

    plt.subplot(3, 1, 1)
    plt.plot(t, y, "b-", linewidth=2)
    plt.grid(True)
    plt.ylabel("amplituda")
    plt.title("u(t) = 1(t)")

    plt.subplot(3, 1, 2)
    plt.plot(t2, y2, "r-", linewidth=2)
    plt.grid(True)
    plt.ylabel("amplituda")
    plt.title("u(t) = 2*1(t)")

    plt.subplot(3, 1, 3)
    plt.plot(t3, y3, "g-", linewidth=2)
    plt.grid(True)
    plt.xlabel("czas [s]")
    plt.ylabel("amplituda")
    plt.title("u(t) = sin(t) - 1/2")

    plt.tight_layout()
    plt.show()


def zad1_1_2():
    R1 = 1
    R2 = 1
    R3 = 1
    C1 = 1
    C2 = 2
    C3 = 3

    A = np.array([[-1 / (R1 * C1), 0, 0], [0, -1 / (R2 * C2), 0], [0, 0, -1 / (R3 * C3)]])
    B = np.array([[1 / (R1 * C1)], [1 / (R2 * C2)], [1 / (R3 * C3)]])
    C = [1, 1, 1]
    D = 0

    # CZYTANIE KILKU ZMIENNYCH STANU I NA JEDNYM WYKRESIE
    # DOPIERO WTEDY MOZNA STWIERDZIC CZY JEST STEROWALNY CZY NIE

    K = np.hstack([B, A @ B, A @ A @ B])
    wiersze = K.shape[0]
    if lin.matrix_rank(K) == wiersze:
        print("Układ oryginalny jest STEROWALNY")
        print(f"rank(K)={lin.matrix_rank(K)}")
        print(f"n={wiersze}")
    else:
        print("Układ oryginalny jest NIESTEROWALNY")
        print(f"rank(K)={lin.matrix_rank(K)}")
        print(f"n={wiersze}")

    system = sp.StateSpace(A, B, C, D)

    t = np.linspace(0, 100, 1000)

    t, y = sp.step(system, T=t)

    u2 = np.ones_like(t) * 2
    t2, y2, _ = sp.lsim(system, U=u2, T=t)

    u3 = np.sin(t) - 0.5
    t3, y3, _ = sp.lsim(system, U=u3, T=t)

    plt.figure(figsize=(10, 10))

    plt.subplot(3, 1, 1)
    plt.plot(t, y, "b-", linewidth=2)
    plt.grid(True)
    plt.ylabel("amplituda")
    plt.title("u(t) = 1(t)")

    plt.subplot(3, 1, 2)
    plt.plot(t2, y2, "r-", linewidth=2)
    plt.grid(True)
    plt.ylabel("amplituda")
    plt.title("u(t) = 2*1(t)")

    plt.subplot(3, 1, 3)
    plt.plot(t3, y3, "g-", linewidth=2)
    plt.grid(True)
    plt.xlabel("czas [s]")
    plt.ylabel("amplituda")
    plt.title("u(t) = sin(t) - 1/2")

    plt.tight_layout()
    plt.show()


def zad1_1_3():
    R = 1
    L = 0.1
    C = 0.1

    A = np.array([[0, 1 / L, 0, 0], [-1 / C, -1 / (R * C), 0, -1 / (R * C)], [0, 0, 0, 1 / L], [0, -1 / (R * C), -1 / C, -1 / (R * C)]])
    B = np.array([[0], [1 / (R * C)], [0], [1 / (R * C)]])
    C = [1, 1, 1, 1]
    D = 0

    K = np.hstack([B, A @ B, A @ A @ B, A @ A @ A @ B])
    wiersze = K.shape[0]
    if lin.matrix_rank(K) == wiersze:
        print("Układ oryginalny jest STEROWALNY")
        print(f"rank(K)={lin.matrix_rank(K)}")
        print(f"n={wiersze}")
    else:
        print("Układ oryginalny jest NIESTEROWALNY")
        print(f"rank(K)={lin.matrix_rank(K)}")
        print(f"n={wiersze}")

    system = sp.StateSpace(A, B, C, D)

    t = np.linspace(0, 100, 1000)

    t, y = sp.step(system, T=t)

    u2 = np.ones_like(t) * 2
    t2, y2, x2 = sp.lsim(system, U=u2, T=t)

    u3 = np.sin(t) - 0.5
    t3, y3, x3 = sp.lsim(system, U=u3, T=t)

    plt.figure(figsize=(10, 10))

    plt.subplot(4, 1, 1)
    plt.plot(t, y, "b-", linewidth=2)
    plt.grid(True)
    plt.ylabel("amplituda")
    plt.title("u(t) = 1(t)")

    plt.subplot(4, 1, 2)
    plt.plot(t2, y2, "r-", linewidth=2)
    plt.grid(True)
    plt.ylabel("amplituda")
    plt.title("u(t) = 2*1(t)")

    plt.subplot(4, 1, 3)
    plt.plot(t3, y3, "g-", linewidth=2)
    plt.grid(True)
    plt.xlabel("czas [s]")
    plt.ylabel("amplituda")
    plt.title("u(t) = sin(t) - 1/2")

    # plt.subplot(4, 1, 4)
    # plt.plot(t3, x3, "g-", linewidth=2)
    # plt.grid(True)
    # plt.xlabel("czas [s]")
    # plt.ylabel("amplituda")
    # plt.title("u(t) = sin(t) - 1/2")

    plt.tight_layout()
    plt.show()


def zad1_1_4():
    R1 = 2
    R2 = 1
    L1 = 0.5
    L2 = 1
    C = 3

    A = np.array([[-R1 / L1, 0, -1 / L1], [0, 0, 1 / L2], [1 / C, -1 / C, -1 / (R2 * C)]])
    B = np.array([[1 / L1], [0], [0]])
    C = [1, 1, 1]
    D = 0

    K = np.hstack([B, A @ B, A @ A @ B])
    wiersze = K.shape[0]

    if lin.matrix_rank(K) == wiersze:
        print("Układ oryginalny jest STEROWALNY")
        print(f"rank(K)={lin.matrix_rank(K)}")
        print(f"n={wiersze}")
    else:
        print("Układ oryginalny jest NIESTEROWALNY")
        print(f"rank(K)={lin.matrix_rank(K)}")
        print(f"n={wiersze}")

    system = sp.StateSpace(A, B, C, D)

    t = np.linspace(0, 100, 1000)

    t, y = sp.step(system, T=t)

    u2 = np.ones_like(t) * 2
    t2, y2, x2 = sp.lsim(system, U=u2, T=t)

    u3 = np.sin(t) - 0.5
    t3, y3, x3 = sp.lsim(system, U=u3, T=t)

    plt.figure(figsize=(10, 10))

    plt.subplot(4, 1, 1)
    plt.plot(t, y, "b-", linewidth=2)
    plt.grid(True)
    plt.ylabel("amplituda")
    plt.title("u(t) = 1(t)")

    plt.subplot(4, 1, 2)
    plt.plot(t2, y2, "r-", linewidth=2)
    plt.grid(True)
    plt.ylabel("amplituda")
    plt.title("u(t) = 2*1(t)")

    plt.subplot(4, 1, 3)
    plt.plot(t3, y3, "g-", linewidth=2)
    plt.grid(True)
    plt.xlabel("czas [s]")
    plt.ylabel("amplituda")
    plt.title("u(t) = sin(t) - 1/2")

    # plt.subplot(4, 1, 4)
    # plt.plot(t3, x3, "g-", linewidth=2)
    # plt.grid(True)
    # plt.xlabel("czas [s]")
    # plt.ylabel("x")
    # plt.title("u(t) = sin(t) - 1/2")

    plt.tight_layout()
    plt.show()


def zad2_1_1():
    C1 = 1
    C2 = 1 / 2
    R1 = 2
    R2 = 4

    A = np.array([[-1 / (R1 * C1), 0], [0, -1 / (R2 * C2)]])
    B = np.array([[1 / (R1 * C1)], [1 / (R2 * C2)]])
    C = [1, 1]
    D = 0

    P_1 = np.hstack([B, A @ B])

    if lin.det(P_1) == 0:
        print("NIE MOZNA WYZNACZYC UKLADU W POSTACI REGULATOROWEJ")
        return

    A1 = lin.inv(P_1) @ A @ P_1
    B1 = lin.inv(P_1) @ B
    C1 = C @ P_1
    D1 = D

    # sys = sp.StateSpace(A1, B1, C1, D1)

    K = np.hstack([B1, A1 @ B1])
    wiersze = K.shape[0]
    if lin.matrix_rank(K) == wiersze:
        print("UKLAD W POSTACI REGULATOROWEJ STEROWALNY")
    else:
        print("UKLAD W POSTACI REGULATOROWEJ NIESTEROWALNY")


def zad2_1_2():
    R1 = 1
    R2 = 2
    R3 = 1
    C1 = 1
    C2 = 2
    C3 = 3

    A = np.array([[-1 / (R1 * C1), 0, 0], [0, -1 / (R2 * C2), 0], [0, 0, -1 / (R3 * C3)]])
    B = np.array([[1 / (R1 * C1)], [1 / (R2 * C2)], [1 / (R3 * C3)]])
    C = [1, 1, 1]
    D = 0

    P_1 = np.hstack([B, A @ B, A @ A @ B])

    if lin.det(P_1) == 0:
        print("NIE MOZNA WYZNACZYC UKLADU W POSTACI REGULATOROWEJ")
        return

    A1 = lin.inv(P_1) @ A @ P_1
    B1 = lin.inv(P_1) @ B
    C1 = C @ P_1
    D1 = D

    # sys = sp.StateSpace(A1, B1, C1, D1)

    K = np.hstack([B1, A1 @ B1, A1 @ A1 @ B1])
    wiersze = K.shape[0]
    if lin.matrix_rank(K) == wiersze:
        print("UKLAD W POSTACI REGULATOROWEJ STEROWALNY")
    else:
        print("UKLAD W POSTACI REGULATOROWEJ NIESTEROWALNY")


def zad2_1_3():
    R = 1
    L = 0.1
    C = 0.1

    A = np.array([[0, 1 / L, 0, 0], [-1 / C, -1 / (R * C), 0, -1 / (R * C)], [0, 0, 0, 1 / L], [0, -1 / (R * C), -1 / C, -1 / (R * C)]])
    B = np.array([[0], [1 / (R * C)], [0], [1 / (R * C)]])
    C = [1, 1, 1, 1]
    D = 0

    P_1 = np.hstack([B, A @ B, A @ A @ B, A @ A @ A @ B])

    if lin.det(P_1) == 0:
        print("NIE MOZNA WYZNACZYC UKLADU W POSTACI REGULATOROWEJ")
        return

    A1 = lin.inv(P_1) @ A @ P_1
    B1 = lin.inv(P_1) @ B
    C1 = C @ P_1
    D1 = D

    # sys = sp.StateSpace(A1, B1, C1, D1)

    K = np.hstack([B1, A1 @ B1, A1 @ A1 @ B1, A1 @ A1 @ A1 @ B1])

    wiersze = K.shape[0]
    if lin.matrix_rank(K) == wiersze:
        print("UKLAD W POSTACI REGULATOROWEJ STEROWALNY")
    else:
        print("UKLAD W POSTACI REGULATOROWEJ NIESTEROWALNY")


def zad2_1_4():
    R1 = 2
    R2 = 1
    L1 = 0.5
    L2 = 1
    C = 3

    A = np.array([[-R1 / L1, 0, -1 / C], [0, 0, 1 / L2], [1 / C, -1 / C, -1 / (R2 * C)]])
    B = np.array([[1 / L1], [0], [0]])
    C = [1, 1, 1]
    D = 0

    P_1 = np.hstack([B, A @ B, A @ A @ B])

    if lin.det(P_1) == 0:
        print("NIE MOZNA WYZNACZYC UKLADU W POSTACI REGULATOROWEJ")
        return

    A1 = lin.inv(P_1) @ A @ P_1
    B1 = lin.inv(P_1) @ B
    C1 = C @ P_1
    D1 = D

    sys = sp.StateSpace(A1, B1, C1, D1)

    K = np.hstack([B1, A1 @ B1, A1 @ A1 @ B1])
    wiersze = K.shape[0]
    if lin.matrix_rank(K) == wiersze:
        print("UKLAD W POSTACI REGULATOROWEJ STEROWALNY")
    else:
        print("UKLAD W POSTACI REGULATOROWEJ NIESTEROWALNY")


def zad2_2():
    R1 = 1
    R2 = 1
    R3 = 1
    C1 = 1
    C2 = 2
    C3 = 3

    A = np.array([[-1 / (R1 * C1), 0, 0], [0, -1 / (R2 * C2), 0], [0, 0, -1 / (R3 * C3)]])
    B = np.array([[1 / (R1 * C1)], [1 / (R2 * C2)], [1 / (R3 * C3)]])
    C = [1, 1, 1]
    D = 0

    system_orig = sp.StateSpace(A, B, C, D)

    P_1 = np.hstack([B, A @ B, A @ A @ B])

    A1 = lin.inv(P_1) @ A @ P_1
    B1 = lin.inv(P_1) @ B
    C1 = C @ P_1
    D1 = D

    rank_P1 = lin.matrix_rank(P_1)
    n = A.shape[0]

    if rank_P1 == n:
        print("UKLAD STEROWALNY")
        print(f"rank={rank_P1}")
        print(f"n={n}")

        A_reg = lin.inv(P_1) @ A @ P_1
        B_reg = lin.inv(P_1) @ B
        C_reg = C @ P_1
        D_reg = D

        system_reg = sp.StateSpace(A_reg, B_reg, C_reg, D_reg)

        t = np.linspace(0, 100, 1000)

        t_orig, y_orig = sp.step(system_orig, T=t)
        t_reg, y_reg = sp.step(system_reg, T=t)

        u3 = np.sin(t) - 0.5
        t3_orig, y3_orig, _ = sp.lsim(system_orig, u3, t)
        t3_reg, y3_reg, _ = sp.lsim(system_reg, u3, t)

        print(f"przebiegi skokowe równe: {np.allclose(y_orig, y_reg)}")
        print(f"przebiegi dla sin: {np.allclose(y3_orig, y3_reg)}")

        plt.figure(figsize=(14, 10))

        plt.subplot(2, 1, 1)
        plt.plot(t_orig, y_orig, "b-", linewidth=4, label="oryginalna")
        plt.plot(t_reg, y_reg, "r--", linewidth=2, label="regularna")
        plt.grid(True)
        plt.xlabel("czas [s]")
        plt.ylabel("amplituda")
        plt.title("odpowiedz skokowa")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(t3_orig, y3_orig, "b-", linewidth=4, label="oryginalna")
        plt.plot(t3_reg, y3_reg, "r--", linewidth=2, label="regularna")
        plt.grid(True)
        plt.xlabel("czas [s]")
        plt.ylabel("amplituda")
        plt.title("odpowiedz na sin(t)-0.5")
        plt.legend()

        plt.tight_layout()
        plt.show()


def zad3_1():
    pass


def zad3_2():
    pass


def zad3_3():
    pass


if __name__ == "__main__":
    # zad1_1_2()
    zad2_2()
