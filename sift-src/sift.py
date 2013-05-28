class Enum(dict):
    """
    Simple class half way between a dict and a class, behaving as an enum
    """
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError

par = Enum("OctaveMax"=100000,
            "DoubleImSize"=0,
            "order"=3,
            "InitSigma"=1.6,
            "BorderDist"=5,
            "Scales"=3,
            "PeakThresh"=255.0 * 0.04 / 3.0,
            "EdgeThresh"=0.06,
            "EdgeThresh1"=0.08,
            "OriBins "=36,
            "OriSigma"=1.5,
            "OriHistThresh"=0.8,
            "MaxIndexVal"=0.2,
            "MagFactor "=3,
            "IndexSigma "=1.0,
            "IgnoreGradSign"=0,
            "MatchRatio"=0.73,
            "MatchXradius"=1000000.0,
            "MatchYradius"=1000000.0,
            "noncorrectlylocalized"=0)