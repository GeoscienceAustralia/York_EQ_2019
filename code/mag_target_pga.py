# find magnitude to match the target PGA at rock
import os
from io import BytesIO
from openquake.hazardlib.geo import Line, Point, Polygon
from openquake.hazardlib.scalerel import WC1994
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib import const
from openquake.hazardlib.geo.surface import SimpleFaultSurface
from openquake.hazardlib.source.rupture import ParametricProbabilisticRupture, BaseRupture

from openquake.hmtk import parsers
from openquake.hazardlib import nrml

from openquake.hazardlib import source, sourceconverter
from openquake.hazardlib.calc.gmf import ground_motion_fields
from openquake.hazardlib.calc.gmf import GmfComputer

from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.imt import SA, PGA, PGV
from openquake.hazardlib.gsim.campbell_bozorgnia_2008 import CampbellBozorgnia2008
from openquake.hazardlib.gsim.akkar_bommer_2010 import AkkarBommer2010
from openquake.hazardlib.gsim.atkinson_boore_2006 import AtkinsonBoore2006
from openquake.hazardlib.gsim.somerville_2009 import SomervilleEtAl2009NonCratonic

from openquake.hazardlib.source import PointSource
from openquake.hazardlib.mfd import ArbitraryMFD
from openquake.hazardlib.scalerel import WC1994, Leonard2014_SCR
from openquake.hazardlib.geo import Point, NodalPlane, Polygon, Line,RectangularMesh
from openquake.hazardlib.geo.surface import PlanarSurface
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.tom import PoissonTOM
#import matplotlib.pyplot as plt
import numpy as np
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from mpl_toolkits.basemap import Basemap
from openquake.hazardlib.geo.geodetic import distance
from openquake.commonlib import readinput

def get_parameters(metadata):

    mags = [float(metadata['Mw'])]
    rupture_aspect_ratio = 1.0
    _lat, _lon = float(metadata['rupture_centroid_lat']), float(metadata['rupture_centroid_lon'])
    _depth = float(metadata['depth'])
    _strike = float(metadata['azimuth'])
    _dip = float(metadata['dip'])
    _rake = float(metadata['rake'])
    # _rake = 90.0 # thrust / reverse type fault
    upper_depth = float(metadata['upper_depth'])
    lower_depth = float(metadata['lower_depth'])
    rupture_mesh_spacing = float(metadata['rupture_mesh_spacing'])

    src = PointSource(
                source_id='1',
                name='point',
                tectonic_region_type="Active Shallow Crust",
                mfd=ArbitraryMFD(magnitudes=mags, occurrence_rates=[1.0]),
                rupture_mesh_spacing=rupture_mesh_spacing,
                magnitude_scaling_relationship=WC1994(),
                rupture_aspect_ratio=rupture_aspect_ratio,
                temporal_occurrence_model=PoissonTOM(50.),
                upper_seismogenic_depth=upper_depth,
                lower_seismogenic_depth=lower_depth,
                location=Point(_lon, _lat),
                nodal_plane_distribution=PMF(
                    [(1.0, NodalPlane(strike=_strike, dip=_dip, rake=_rake))]),
                hypocenter_distribution=PMF([(1.0, _depth)])
    )

    for rup in src.iter_ruptures():
        surf = rup.surface

    for att in ['bottom_left', 'bottom_right', 'top_left', 'top_right']:
        _instance = getattr(surf, att)
        for att2 in ['latitude', 'longitude', 'depth']:
            key = '{}_{}'.format(att, att2)
            metadata[key] = '{}'.format(getattr(_instance, att2))

    metadata['rake'] = '{}'.format(_rake)

    return metadata

def estimate_PGA_given_mag_depth(metadata_dic, ini_file, flag=False):

    assert os.path.exists(ini_file)
    oqparam = readinput.get_oqparam(ini_file)

    # metadata for WACA
    # metadata = {'Mw': mag, 
    #             'depth': depth,
    #             'rupture_centroid_lat': -31.953, 
    #             'rupture_centroid_lon': 115.88,
    #             'dip': 45.0, 
    #             'azimuth': 0.0, 
    #             'length': 1.0,
    #             'width': 1.0,
    #             'upper_depth': 1.0,
    #             'lower_depth': 50.0}

    # MUN
    # metadata = {'Mw': mag, 
    #             'depth': depth,
    #             'rupture_centroid_lat': rupture_centroid[1], 
    #             'rupture_centroid_lon': rupture_centroid[0],
    #             'dip': 45.0, 
    #             'azimuth': 0.0, 
    #             'length': 1.0,
    #             'width': 1.0,
    #             'upper_depth': upper_depth,
    #             'lower_depth': 50.0}

    metadata = get_parameters(metadata_dic)

    _str = """\
<?xml version="1.0" encoding="utf-8"?>
<nrml xmlns:gml="http://www.opengis.net/gml" xmlns="http://openquake.org/xmlns/nrml/0.5"> 
    <singlePlaneRupture>
        <magnitude>{Mw}</magnitude>
        <rake>{rake}</rake>
        <hypocenter lat="{rupture_centroid_lat}" lon="{rupture_centroid_lon}" depth="{depth}"/>
            <planarSurface strike="{azimuth}" dip="{dip}">
                <topLeft lon="{top_left_longitude}" lat="{top_left_latitude}" depth="{top_left_depth}"/>
                <topRight lon="{top_right_longitude}" lat="{top_right_latitude}" depth="{top_right_depth}"/>
                <bottomLeft lon="{bottom_left_longitude}" lat="{bottom_left_latitude}" depth="{bottom_left_depth}"/>
                <bottomRight lon="{bottom_right_longitude}" lat="{bottom_right_latitude}" depth="{bottom_right_depth}"/>
            </planarSurface>
    </singlePlaneRupture>
</nrml>
""".format(**metadata)
    xml_file = BytesIO(str.encode(_str))

    #xml_file = '/Users/hyeuk/Projects/oq_exercise/xxx/rupture_model.xml'
    converter = sourceconverter.RuptureConverter(rupture_mesh_spacing=2.0, 
                                                complex_fault_mesh_spacing=1.5)
    [node] = nrml.read(xml_file)
    rupture = converter.convert_node(node)

    # rupture.tectonic_region_type

    # site for WACA
    # site1 = Site(location=Point(115.8739, -31.9611), vs30=760.,
    #             vs30measured=True, z1pt0=41.3066, z2pt5=0.6673)

    # site for MUN
    # site_lon, site_lat = 
    # site1 = Site(location=Point(site_lon, site_lat), vs30=760.,
    #             vs30measured=True, z1pt0=41.3066, z2pt5=0.6673)
    # site2 = Site(location=Point(site_lon, site_lat), vs30=760.,
    #             vs30measured=True, z1pt0=100.0, z2pt5=0.0001)

    # sites = SiteCollection([site1, site2])
    # sites = SiteCollection([site1])
    site = readinput.get_site_collection(oqparam)

    # imts = [PGA()]
    # gc = GmfComputer(rupture, sites, [str(imt) for imt in imts], gsim,
    #                  truncation_level=0, correlation_model=None)

    # realizations = 1
    # res = gc.compute(gsim, realizations, seed=None)

    # oqparam = readinput.get_oqparam('/Users/hyeuk/Projects/scenario_Perth/NSHA18/job_hazard_scenario.ini')
    gsim_lt = readinput.get_gsim_lt(oqparam)                                                              

    value = 0.0
    for x in gsim_lt:
        fields = ground_motion_fields(
            rupture=rupture,
            sites=readinput.get_site_collection(oqparam),
            imts=readinput.get_imts(oqparam),
            gsim=x.value[0],
            truncation_level=0,
            realizations=1,
        )
        _pga = fields[PGA()]
        print('{}->{},{}'.format(x.value[0], _pga, x.weight['weight']))
        value += float(x.weight['weight']) * _pga

    if flag:   
        with open('tmp.xml', 'w') as out:
            out.write(_str)

    return value

# Rp500
# |Calingiri| 5.03|15| 5-5.2|
# Calingiri': {1: ('d10', 'lo24', 'la20', 191.0, 14.546839299314547
# # lo24: 116.6-116.8
# # 1a20: -31.9 -31.7
# sites = 116.7683 -31.8912
# - PGA at 500 RP: 5.935e-2
metadata_dic = {
    'Mw': 5.03, 
    'depth': 15.0,
    # 'rupture_centroid_lat': -31.77327, 
    # 'rupture_centroid_lat': -31.7, 
    'rupture_centroid_lat': -31.755, 
    # 'rupture_centroid_lon': 116.68339,
    #'rupture_centroid_lon': 116.6,
    'rupture_centroid_lon': 116.65,
    'dip': 35.0, 
    'azimuth': 178.0, 
    'rake': 90.0,
    'rupture_mesh_spacing': 2.0,
    'upper_depth': 1.0,
    'lower_depth': 50.0}

# Rp1000
#|Lake Muir | 5.3 | 2 | 5.2-5.4|
# lat="-3.1720373E+01" lon="1.1695354E+02"/>
# sites = 116.7683 -31.8912
#- PGA at 1000 RP: 1.0203e-1
# lo25: 116.8-117.0
# 1a20: -31.9 -31.7
# 
metadata_dic = {
    'Mw': 5.3, 
    'depth': 2.0,
    'rupture_centroid_lat': -31.8204, 
    'rupture_centroid_lon': 116.9335,
    'dip': 35.538, 
    'azimuth': 177.9486, 
    'rake': 90.0,
    'rupture_mesh_spacing': 2.0,
    'upper_depth': 1.0,
    'lower_depth': 50.0}

# Rp2500
# |Meckering| 6.58| 10| 6.4-6.6|
# lo26: 117.0-117.2
# la19: -32.1 -31.9
# sites = 116.7683 -31.8912
#- PGA at 2500 RP: 1.9875e-1

metadata_dic = {
    'Mw': 6.58, 
    'depth': 10.0,
    'rupture_centroid_lat': -31.9057, 
    'rupture_centroid_lon': 117.0571,
    'dip': 35.0, 
    'azimuth': 178.972, 
    'rake': 90.0,
    'rupture_mesh_spacing': 2.0,
    'upper_depth': 1.0,
    'lower_depth': 50.0}

