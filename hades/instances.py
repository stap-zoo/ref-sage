from fields import BN254_SCALAR, BLS12_381_SCALAR, ST, GOLDILOCKS, MERSENNE31

from hades.params import PoseidonParams, Poseidon2Params, NeptuneParams

# Parameters mirror the reference implementations:
#   - Poseidon, Poseidon2, Neptune (ISEC): https://extgit.isec.tugraz.at/krypto/zkfriendlyhashzoo
# Further implementations:
#   - Poseidon2 (HorizenLabs), https://github.com/HorizenLabs/poseidon2
#   - Poseidon (Ethereum), https://github.com/khovratovich/poseidon-tools
#   - Poseidon (Circom), https://github.com/iden3/circomlib
# Round constants and matrices are derived by the construction strategies in params.py
# (verified bit-for-bit against the reference constants in tests); the Poseidon2 internal-matrix
# diagonals (MAT_DIAG_M_1), which are field-specific and not formulaic, are supplied here. Each
# instance is named <PRIMITIVE>_<FIELD>_T<state size>. The sponge parameters (rate r, capacity c,
# digest size d) are required and given explicitly; these use the standard r = t-1, c = 1, d = 1.


# ---------------------------------------------------------------------------
# Poseidon
# ---------------------------------------------------------------------------
# version="isec" matches the IAIK/zkhashzoo reference; the BLS12-381
# t=2 instance was generated circom-style (s-box marker 0).

POSEIDON_BN254_T3      = PoseidonParams(p=BN254_SCALAR.p,      t=3,  alpha=5, R_ext=8, R_int=57, r=2,  c=1, d=1)
POSEIDON_BLS12_T3      = PoseidonParams(p=BLS12_381_SCALAR.p,  t=3,  alpha=5, R_ext=8, R_int=57, r=2,  c=1, d=1)
POSEIDON_BLS12_T2      = PoseidonParams(p=BLS12_381_SCALAR.p,  t=2,  alpha=5, R_ext=8, R_int=56, r=1,  c=1, d=1, version="circom")
POSEIDON_ST_T3         = PoseidonParams(p=ST.p,                t=3,  alpha=3, R_ext=8, R_int=84, r=2,  c=1, d=1)
POSEIDON_GOLDILOCKS_T8 = PoseidonParams(p=GOLDILOCKS.p,        t=8,  alpha=7, R_ext=8, R_int=22, r=7,  c=1, d=1)
POSEIDON_GOLDILOCKS_T12 = PoseidonParams(p=GOLDILOCKS.p,       t=12, alpha=7, R_ext=8, R_int=22, r=11, c=1, d=1)
POSEIDON_MERSENNE_T16  = PoseidonParams(p=MERSENNE31.p,        t=16, alpha=5, R_ext=8, R_int=14, r=15, c=1, d=1)
POSEIDON_MERSENNE_T24  = PoseidonParams(p=MERSENNE31.p,        t=24, alpha=5, R_ext=8, R_int=22, r=23, c=1, d=1)


# ---------------------------------------------------------------------------
# Poseidon2
# ---------------------------------------------------------------------------

POSEIDON2_BLS12_T2 = Poseidon2Params(p=BLS12_381_SCALAR.p, t=2, alpha=5, R_ext=8, R_int=56, r=1, c=1, d=1,
                                     mat_diag=[1, 2])
POSEIDON2_BLS12_T3 = Poseidon2Params(p=BLS12_381_SCALAR.p, t=3, alpha=5, R_ext=8, R_int=56, r=2, c=1, d=1,
                                     mat_diag=[1, 1, 2])
POSEIDON2_BLS12_T4 = Poseidon2Params(p=BLS12_381_SCALAR.p, t=4, alpha=5, R_ext=8, R_int=56, r=3, c=1, d=1, mat_diag=[
    1655454839116271620575749293234574274222993456067245397034476501783047702986,
    50438009192746386491573599403504532375212735259047687062053082186860235946862,
    2333904567228711547445735898465837165392237851337731850779276628672040817654,
    50596178259748710644710488613201561091625758003284944221322227344704022331809])
POSEIDON2_BLS12_T8 = Poseidon2Params(p=BLS12_381_SCALAR.p, t=8, alpha=5, R_ext=8, R_int=57, r=7, c=1, d=1, mat_diag=[
    30096150855626815013648098211149190726024462037711422151903740709493260300796,
    1670165440712639538351788804048794320334719999221164772595439843145725878342,
    46650283196255499715295956296451823074186598793282238922518499397091394504987,
    35540998704038434561306263855447791582438082003486383019176839889144643735624,
    5729993341210689688259865343737097561899201100829809250544373980467298239994,
    50448381164981534770780393450590297246005426700091703154028953045831153438287,
    17212607781899146579730953203574521410127157101393873991200310682506890698706,
    34706517679585386573351886038051072526558536116546116247060202675936040458533])
POSEIDON2_BN254_T3 = Poseidon2Params(p=BN254_SCALAR.p, t=3, alpha=5, R_ext=8, R_int=56, r=2, c=1, d=1,
                                     mat_diag=[1, 1, 2])

POSEIDON2_GOLDILOCKS_T8 = Poseidon2Params(p=GOLDILOCKS.p, t=8, alpha=7, R_ext=8, R_int=22, r=7, c=1, d=1, mat_diag=[
    15382945929615919109, 12259423718610140002, 2923504536469986317, 4443807569370001966,
    16558326539122289762, 5504310531611560966, 2280505357293205149, 11068236061408781395])
POSEIDON2_GOLDILOCKS_T12 = Poseidon2Params(p=GOLDILOCKS.p, t=12, alpha=7, R_ext=8, R_int=22, r=11, c=1, d=1, mat_diag=[
    14947297269260626681, 4599513150284541628, 11201225350677596328, 16470806799883061954,
    7039824439425940594, 3386007053668868578, 9623872391379616577, 3286537768282113302,
    5845694577252884219, 7132727139459816539, 11806562510236171986, 11024094583410641203])
POSEIDON2_GOLDILOCKS_T16 = Poseidon2Params(p=GOLDILOCKS.p, t=16, alpha=7, R_ext=8, R_int=22, r=15, c=1, d=1, mat_diag=[
    7235003234611667288, 13607748933422710824, 9679244401895375471, 16479082478743854625,
    3289023727303324218, 4787374403950870133, 8891042382918883624, 15561457801946963383,
    5603344928999172045, 16742247955352235918, 17266655938423716472, 12371113468785829922,
    668040572637163662, 4795111821436112645, 13703345978056153033, 5190976229883216617])
POSEIDON2_GOLDILOCKS_T20 = Poseidon2Params(p=GOLDILOCKS.p, t=20, alpha=7, R_ext=8, R_int=22, r=19, c=1, d=1, mat_diag=[
    18095720882400488028, 2897395537603417490, 3899157398245026194, 15999195304901058362,
    3411690334970111278, 2818673194506799940, 17596225219062753602, 12859877527836743214,
    6713417324070002009, 5976567677901547227, 6262938817599225258, 272194192324957864,
    2778978430777956109, 17430330283560822270, 13313823783807753918, 6038613023506328008,
    16873383611658653480, 9586241208828787579, 6270886117298114176, 13558493551369839869])

POSEIDON2_MERSENNE_T16 = Poseidon2Params(p=MERSENNE31.p, t=16, alpha=5, R_ext=8, R_int=14, r=15, c=1, d=1, mat_diag=[
    1146088468, 1460352161, 1260749089, 238394597, 743094189, 693825473, 1841182985, 1101428780,
    2073797751, 1484722434, 536468193, 890031844, 1468529595, 1335340556, 336328943, 1817285350])
POSEIDON2_MERSENNE_T24 = Poseidon2Params(p=MERSENNE31.p, t=24, alpha=5, R_ext=8, R_int=22, r=23, c=1, d=1, mat_diag=[
    2039490307, 2093221626, 1711511959, 293966760, 792113904, 2144084959, 405387875, 1904256531,
    1676399337, 293135707, 425672878, 756903248, 1211773568, 639051394, 918782105, 967362913,
    1596177038, 1222601859, 953579434, 564145768, 946671201, 1776634772, 809235858, 2016995784])


# ---------------------------------------------------------------------------
# Neptune
# ---------------------------------------------------------------------------
# All constants/matrices are derived deterministically from SHAKE128 (no spec data to supply).

NEPTUNE_BN254_T4       = NeptuneParams(p=BN254_SCALAR.p,     t=4,  alpha=5, R_ext=6, R_int=68, r=3,  c=1, d=1)
NEPTUNE_BLS12_T4       = NeptuneParams(p=BLS12_381_SCALAR.p, t=4,  alpha=5, R_ext=6, R_int=68, r=3,  c=1, d=1)
NEPTUNE_BLS12_T2       = NeptuneParams(p=BLS12_381_SCALAR.p, t=2,  alpha=5, R_ext=8, R_int=56, r=1,  c=1, d=1)
NEPTUNE_ST_T4          = NeptuneParams(p=ST.p,               t=4,  alpha=3, R_ext=6, R_int=96, r=3,  c=1, d=1)
NEPTUNE_GOLDILOCKS_T8  = NeptuneParams(p=GOLDILOCKS.p,       t=8,  alpha=7, R_ext=6, R_int=38, r=7,  c=1, d=1)
NEPTUNE_GOLDILOCKS_T12 = NeptuneParams(p=GOLDILOCKS.p,       t=12, alpha=7, R_ext=6, R_int=42, r=11, c=1, d=1)
