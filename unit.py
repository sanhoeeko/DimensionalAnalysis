"""
Order of basic SI units:
    s   [T]
    m   [L]
    kg  [M]
    A   [I]
    K   [K]
    mol [N]
"""

import numpy as np
import fractions as fr
import re

dic = 'TLMIKN'

# physical constants
_c = 299792458  # meters per second, the speed of light in a vacuum
_hbar = 1.054571817e-34  # Joule seconds
_e = 1.602176634e-19  # Coulombs, the charge of an electron
_kB = 1.380649e-23  # Joules per Kelvin, the Boltzmann constant
_nA = 6.02214076e23  # inverse mol, the Avogadro constant
_g = 6.6743e-11  # gravity constant


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
    def __init__(self, name, unit, value):
        self.name = name
        self.unit = unit
        self.value = value

    def __repr__(self):
        return self.name + ' = ' + str(self.value) + ' ' + str(self.unit)


class UnitSpace:
    def __init__(self, constants: list[Constant]):
        self.mat = np.vstack(list(map(lambda x: x.unit.vec, constants)))
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


class IUnit(Unit):
    def __init__(self, vec: np.ndarray, base: UnitSpace):
        super(IUnit, self).__init__(vec)
        self.base = base

    def __mul__(self, o: 'IUnit'):
        if self.base != o.base:
            raise ValueError
        return IUnit(self.vec + o.vec, self.base)

    def __truediv__(self, o: 'IUnit'):
        if self.base != o.base:
            raise ValueError
        return IUnit(self.vec - o.vec, self.base)

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
Hamilton = M * Speed ** 2  # energy
Action = Hamilton * T  # action / angular momentun
Area = L ** 2  # area
Volume = L ** 3  # volume
Dense = Volume ** -1  # particle density
Gravity = L ** 3 * M ** -1 * T ** -2

# ED:
Charge = I / Speed / Area / Dense  # charge

# SM:
Temp = K  # temperature
Boltzmann = Hamilton / Temp  # kB
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

PlanckUnit = UnitSpace([c_, hbar_, G_, e_, kB_, nA_])
NaturalMassUnit = UnitSpace([Constant('M', M, 1), c_, hbar_, e_, kB_, nA_])
NaturalLengthUnit = UnitSpace([Constant('L', L, 1), c_, hbar_, e_, kB_, nA_])


if __name__ == '__main__':

    # convert the well-known static energy 511keV to the mass of a electron

    u = NaturalMassUnit.value_convert(Constant('E', Hamilton, 511e3*_e))
    print(u)
