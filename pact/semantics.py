"""Natural-language descriptions of the benchmark datasets and their variables (for the semantic prompt).

Descriptions follow the dataset documentation (Informer for ETT, Autoformer for Weather/ILI/ECL).
"""

DOMAIN = {
    "ETTh1": "This is hourly data from an electricity transformer at a station in China.",
    "ETTh2": "This is hourly data from an electricity transformer at a second station in China.",
    "ETTm1": "This is 15-minute data from an electricity transformer at a station in China.",
    "ETTm2": "This is 15-minute data from an electricity transformer at a second station in China.",
    "Weather": "This is 10-minute weather data recorded at the Max Planck Institute for Biogeochemistry in Jena, Germany.",
    "ILI": "This is weekly influenza-like illness surveillance data from the US Centers for Disease Control and Prevention.",
    "ECL": "This is hourly electricity consumption of clients of a Portuguese utility.",
    "Traffic": "This is hourly road occupancy measured by sensors on San Francisco Bay Area freeways.",
    "Solar": "This is solar power produced by photovoltaic plants in Alabama, recorded every 10 minutes.",
    "PEMS08": "This is traffic flow measured every 5 minutes by detectors on freeways in San Bernardino, California.",
}

_ETT = {
    "HUFL": "the high useful load of the transformer",
    "HULL": "the high useless load of the transformer",
    "MUFL": "the middle useful load of the transformer",
    "MULL": "the middle useless load of the transformer",
    "LUFL": "the low useful load of the transformer",
    "LULL": "the low useless load of the transformer",
    "OT": "the oil temperature of the transformer",
}

_WEATHER = {
    "p (mbar)": "the air pressure in millibar",
    "T (degC)": "the air temperature in degrees Celsius",
    "Tpot (K)": "the potential temperature in Kelvin",
    "Tdew (degC)": "the dew point temperature in degrees Celsius",
    "rh (%)": "the relative humidity in percent",
    "VPmax (mbar)": "the saturation vapour pressure in millibar",
    "VPact (mbar)": "the actual vapour pressure in millibar",
    "VPdef (mbar)": "the vapour pressure deficit in millibar",
    "sh (g/kg)": "the specific humidity in grams per kilogram",
    "H2OC (mmol/mol)": "the water vapour concentration in millimoles per mole",
    "rho (g/m**3)": "the air density in grams per cubic metre",
    "wv (m/s)": "the wind velocity in metres per second",
    "max. wv (m/s)": "the maximum wind velocity in metres per second",
    "wd (deg)": "the wind direction in degrees",
    "rain (mm)": "the rainfall in millimetres",
    "raining (s)": "the duration of rain in seconds",
    "SWDR": "the short-wave downward radiation",
    "PAR": "the photosynthetically active radiation",
    "max. PAR": "the maximum photosynthetically active radiation",
    "Tlog (degC)": "the internal logger temperature in degrees Celsius",
    "OT": "the carbon dioxide concentration in parts per million",
}

_ILI = {
    "% WEIGHTED ILI": "the weighted percentage of patient visits for influenza-like illness",
    "%UNWEIGHTED ILI": "the unweighted percentage of patient visits for influenza-like illness",
    "AGE 0-4": "the number of influenza-like illness patients aged 0 to 4",
    "AGE 5-24": "the number of influenza-like illness patients aged 5 to 24",
    "ILITOTAL": "the total number of influenza-like illness patients",
    "NUM. OF PROVIDERS": "the number of reporting healthcare providers",
    "OT": "the total number of patient visits",
}


def describe(dataset, column, index):
    """Return a phrase describing one variable."""
    if dataset.startswith("ETT"):
        return _ETT.get(column, f"variable {column}")
    if dataset == "Weather":
        for key, text in _WEATHER.items():  # radiation columns contain non-ASCII unit symbols
            if column == key or column.startswith(key + " "):
                return text
        return f"variable {column}"
    if dataset == "ILI":
        return _ILI.get(column, f"variable {column}")
    if dataset == "ECL":
        return f"the electricity consumption of client {index + 1}"
    if dataset == "Traffic":
        return f"the occupancy rate of road sensor {index + 1}"
    if dataset == "Solar":
        return f"the power output of photovoltaic plant {index + 1}"
    if dataset == "PEMS08":
        return f"the traffic flow at detector {index + 1}"
    return f"variable {column}"


def variable_sentence(dataset, column, index):
    return f"{DOMAIN.get(dataset, '')} The variable is {describe(dataset, column, index)}.".strip()
