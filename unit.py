"""
Order of basic SI units:
    s   [T]
    m   [L]
    kg  [M]
    A   [I]
    K   [K]
    mol [N]
"""

import fractions as fr
import re

import numpy as np

dic = 'TLMIKN'

# physical constants
_c = 299792458  # meters per second, the speed of light in a vacuum
_hbar = 1.054571817e-34  # Joule seconds
_e = 1.602176634e-19  # Coulombs, the charge of an electron
_kB = 1.380649e-23  # Joules per Kelvin, the Boltzmann constant
_nA = 6.02214076e23  # inverse mol, the Avogadro constant
_g = 6.6743e-11  # gravity constant
_me = 9.10938356e-31  # kilograms, the mass of an electron


class Unit:
    def __init__(self, vec: np.ndarray):
        self.vec = vec.astype(float)

    def __mul__(self, o: 'Unit'):
        return Unit(self.vec + o.vec)

    def __truediv__(self, o: 'Unit'):
        return Unit(self.vec - o.vec)

    def __pow__(self, power: int):
        return Unit(self.vec * int(power))

    def __repr__(self):
        res = []
        for i in range(6):
            if fr.Fraction(self.vec[i]).limit_denominator() == 0: continue
            s = '[' + dic[i] + ']' + str(fr.Fraction(self.vec[i]).limit_denominator())
            res.append(s)
        return ' '.join(res)


def unit_from_str(string: str):
    u = Unit(np.zeros((6,)))
    arr = string.split(' ')
    for a in arr:
        letter = re.findall(r'\[([A-Z])\]', a)
        integer = a[3:]
        if letter:
            u.vec[dic.index(letter[0])] = int(integer) if len(integer) else 1
    return u


def basic_unit(idx: int):
    u = Unit(np.zeros((6,)))
    u.vec[idx] = 1
    return u


class Constant:
    def __init__(self, name, unit, value=1.0):
        self.name = name
        self.unit = unit
        self.value = value

    def __mul__(self, o: 'Constant'):
        return Constant('', self.unit * o.unit, self.value * o.value)

    def __truediv__(self, o: 'Constant'):
        return Constant('', self.unit / o.unit, self.value / o.value)

    def __pow__(self, power: int):
        return Constant('', self.unit ** power, self.value ** power)

    def __repr__(self):
        return self.name + ' = ' + str(self.value) + ' ' + str(self.unit)


"""
example:

    harmonic = UnitSpace([hbar_, Constant('m',M), Constant('w',T**-1)])
    print(harmonic.unit_convert(L))

"""


class UnitSpace:
    def __init__(self, constants: list[Constant]):
        mat = np.vstack(list(map(lambda x: x.unit.vec, constants)))
        m, n = mat.shape
        if m == n:
            self.mat = mat
        else:
            omitted = np.all(mat == 0, axis=0)
            if np.sum(
                    omitted) != n - m: raise ValueError  # that is because of too many or few constants to form a space
            self.mat = np.eye(n)
            cnt = 0
            for i in range(n):
                if not omitted[i]:
                    self.mat[i, :] = mat[cnt, :]
                    cnt += 1
        self.U = np.linalg.inv(self.mat)
        self.dic = list(map(lambda x: x.name, constants))
        self.ccs = np.array(list(map(lambda x: x.value, constants)))  # means "convert constants"

    def __eq__(self, other):
        return (self.mat == other.mat).all()

    def unit_convert(self, u: Unit):
        return IUnit(u.vec @ self.U, self)

    def value_convert(self, co: Constant):
        new_unit = self.unit_convert(co.unit)
        convert_constant = np.prod(self.ccs ** new_unit.vec)
        new_val = co.value / convert_constant
        return Constant(co.name, new_unit, new_val)

    def factor(self, si_unit: Unit):
        res = self.value_convert(Constant('', si_unit))
        res.value **= -1
        return res

    def unit_to(self, origin: Unit, target: Unit):
        """
        show that what is the factor, that Origin * [Factor] = Target
        """
        factor = target / origin
        return self.unit_convert(factor)

    def value_to(self, origin: Constant, target: Unit):
        factor_unit = target / origin.unit
        factor = self.factor(factor_unit)
        return origin * factor


class IUnit(Unit):
    def __init__(self, vec: np.ndarray, base: UnitSpace):
        super(IUnit, self).__init__(vec)
        self.base = base

    def si(self):
        return Unit(self.vec @ self.base.mat)

    def __mul__(self, o: 'IUnit'):
        if self.base != o.base:
            raise ValueError
        return IUnit(self.vec + o.vec, self.base)

    def __rmul__(self, o: Unit):
        return o * self.si()

    def __truediv__(self, o: 'IUnit'):
        if self.base != o.base:
            raise ValueError
        return IUnit(self.vec - o.vec, self.base)

    def __rdiv__(self, o: Unit):
        return o / self.si()

    def __repr__(self):
        res = []
        for i in range(6):
            if fr.Fraction(self.vec[i]).limit_denominator() == 0: continue
            s = '[' + self.base.dic[i] + ']' + str(fr.Fraction(self.vec[i]).limit_denominator())
            res.append(s)
        return ' '.join(res)


# basic units
T = basic_unit(0)
L = basic_unit(1)
M = basic_unit(2)
I = basic_unit(3)
K = basic_unit(4)
N = basic_unit(5)

"""
induced units
"""
# CM & QM:
Speed = L / T  # speed
Frequency = T ** -1  # frequency
Hamilton = M * Speed ** 2  # energy
Action = Hamilton * T  # action / angular momentum
Area = L ** 2  # area
Volume = L ** 3  # volume
Dense = Volume ** -1  # particle density
Momentum = M * Speed  # momentum
Force = Momentum / T  # force
Gravity = L ** 3 * M ** -1 * T ** -2

# ED:
Charge = I / Speed / Area / Dense  # charge
Upotential = Hamilton / I  # electric potential
Efield = Upotential / L  # electric field
Dfield = Hamilton / Efield / Volume  # polarization field
Bfield = Force / I / L  # magnetic field
Mfield = Hamilton / Bfield / Volume  # magnetic polarization field

# SM:
Temp = K  # temperature
Entropy = Hamilton / Temp  # entropy
Boltzmann = Entropy / Temp  # kB
Pressure = Hamilton / Volume  # pressure
ChemicalPotential = Hamilton / Dense  # chemical potential
HeatCapacity = Boltzmann  # heat capacity
NA = N ** -1

"""
Planck units
"""
c_ = Constant('c', Speed, _c)
hbar_ = Constant('hbar', Action, _hbar)
G_ = Constant('G', Gravity, _g)
e_ = Constant('e', Charge, _e)
kB_ = Constant('kB', Boltzmann, _kB)
nA_ = Constant('NA', NA, _nA)
me_ = Constant('me', M, _me)

PlanckUnit = UnitSpace([c_, hbar_, G_, e_, kB_, nA_])
NaturalM = UnitSpace([Constant('M', M, 1), c_, hbar_, e_, kB_, nA_])
NaturalL = UnitSpace([Constant('L', L, 1), c_, hbar_, e_, kB_, nA_])


class Energy:
    def __init__(self, si: float):
        self.si = Constant('', Hamilton, si)

    def ev(self):
        return self.si.value / _e

    def k(self):
        return NaturalL.value_convert(self.si)

    def wavelength(self):
        return 1 / self.k().value

    def freq(self):
        return NaturalM.value_to(origin=self.si, target=Frequency)

    def mass(self):
        return NaturalM.value_to(origin=self.si, target=M)


def energy_from_ev(ev: float):
    return Energy(ev * _e)


def energy_from_wavelength(lamda: float):
    lamda = Constant('', L ** -1, 1 / lamda)
    joule = NaturalL.value_to(origin=lamda, target=Hamilton).value
    return Energy(joule)


def energy_from_freq(freq: float):
    freq = Constant('', Frequency, freq)
    joule = NaturalM.value_to(origin=freq, target=Hamilton).value
    return Energy(joule)


def energy_from_mass(mass: float):
    mass = Constant('', M, mass)
    joule = NaturalM.value_to(origin=mass, target=Hamilton).value
    return Energy(joule)


if __name__ == '__main__':
    test = energy_from_mass(_me)
    print(test.ev())
