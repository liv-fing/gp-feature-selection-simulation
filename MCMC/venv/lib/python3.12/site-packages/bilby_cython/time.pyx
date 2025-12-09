import numpy as np
from cmath import pi
from cpython.datetime cimport datetime, timedelta
cimport cython

cdef datetime GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)
cdef double EPOCH_J2000_0_JD = 2451545.0
cdef long[18] LEAP_SECONDS = [
    46828800,
    78364801,
    109900802,
    173059203,
    252028804,
    315187205,
    346723206,
    393984007,
    425520008,
    457056009,
    504489610,
    551750411,
    599184012,
    820108813,
    914803214,
    1025136015,
    1119744016,
    1167264017,
]
cdef int NUM_LEAPS = 18

ctypedef fused real:
    short
    int
    long
    float
    double
    long double


@cython.ufunc
cdef int n_leap_seconds(real date):
    """
    Find the number of leap seconds required for the specified date.

    Search in reverse order as in practice, almost all requested times will
    be after the most recent leap.
    """
    cdef int n_leaps = NUM_LEAPS
    if date > LEAP_SECONDS[NUM_LEAPS - 1]:
        return NUM_LEAPS
    while (n_leaps > 0) and (date < LEAP_SECONDS[n_leaps - 1]):
        n_leaps -= 1
    return n_leaps


@cython.ufunc
cdef datetime gps_time_to_utc(real secs):
    """
    Convert from GPS time to UTC, this is a necessary intermediate step in
    converting from GPS to GMST.

    Add the number of seconds minus the leap seconds to the GPS epoch.

    Parameters
    ----------
    secs: double
        The time to convert in GPS time.
    """
    cdef datetime date_before_leaps
    date_before_leaps = GPS_EPOCH + timedelta(seconds=<double>secs - n_leap_seconds(secs))
    return date_before_leaps


@cython.ufunc
cdef double utc_to_julian_day(datetime time):
    """
    Convert from UTC to Julian day, this is a necessary intermediate step in
    converting from GPS to GMST.

    Parameters
    ----------
    time: datetime
        The UTC time to convert
    """
    cdef int year = time.year
    cdef int month = time.month
    cdef int day = time.day
    cdef double second = time.second + 60 * (time.minute + 60 * time.hour)
    return (
        367 * year
        - 7 * (year + (month + 9) / 12) // 4
        + 275 * month // 9
        + day
        + 1721014
        + second / (60 * 60 * 24)
        - 0.5
    )


@cython.ufunc
cdef double greenwich_mean_sidereal_time(real gps_time):
    """
    Compute the Greenwich mean sidereal time from the GPS time.

    Parameters
    ----------
    gps_time: double
        The GPS time to convert
    """
    return greenwich_sidereal_time(gps_time, 0)


@cython.ufunc
cdef double greenwich_sidereal_time(real gps_time, real equation_of_equinoxes):
    """
    Compute the Greenwich mean sidereal time from the GPS time and equation of
    equinoxes.

    Based on XLALGreenwichSiderealTime in lalsuite/lal/lib/XLALSiderealTime.c.

    Parameters
    ----------
    gps_time: double
        The GPS time to convert
    equation_of_equinoxes: double
        The equation of equinoxes
    """
    cdef double julian_day, t_hi, t_lo, t, sidereal_time
    julian_day = utc_to_julian_day(gps_time_to_utc(gps_time))
    t_hi = (julian_day - EPOCH_J2000_0_JD) / 36525.0
    t_lo = (gps_time % 1) / (36525.0 * 86400.0)

    t = t_hi + t_lo

    sidereal_time = equation_of_equinoxes + (-6.2e-6 * t + 0.093104) * t**2 + 67310.54841
    sidereal_time += 8640184.812866 * t_lo
    sidereal_time += 3155760000.0 * t_lo
    sidereal_time += 8640184.812866 * t_hi
    sidereal_time += 3155760000.0 * t_hi

    return sidereal_time * pi / 43200.0
