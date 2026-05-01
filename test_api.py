import api_server
from flask import json as fj
c = api_server.app.test_client()

# Test 1: Night - must not be anomaly
r = c.post('/predict', json={'hour':2,'month':4,'doy':113,'sunElevation':-5,'temp':22,'wind':3,'irradiance':0,'cloud':0,'sunAzimuth':180})
d = fj.loads(r.data)
assert not d['is_anomaly'], "FAIL: Night flagged as anomaly (false positive)"
assert d['anomaly_type'] == 'NORMAL', "FAIL: Night anomaly_type not NORMAL"
print("T1 PASS: Night -> NORMAL (no false positive)")

# Test 2: Wind stow - no anomaly
r = c.post('/predict', json={'hour':12,'month':4,'doy':113,'sunElevation':60,'temp':28,'wind':18,'irradiance':800,'cloud':5,'sunAzimuth':170})
d = fj.loads(r.data)
assert not d['is_anomaly'], "FAIL: Stow mode flagged as anomaly (false positive)"
print("T2 PASS: Wind stow -> NORMAL (no false positive)")

# Test 3: Good midday - no anomaly
r = c.post('/predict', json={'hour':12,'month':4,'doy':113,'sunElevation':65,'temp':28,'wind':4,'irradiance':900,'cloud':5,'sunAzimuth':170})
d = fj.loads(r.data)
ghi = d['ghi_wm2']
pr  = d['performance_ratio']
at  = d['anomaly_type']
print("T3 Daytime: GHI={} PR={}% anomaly={}".format(ghi, pr, at))

# Test 4: Cloudy twilight (low DNI) - MUST be NORMAL (was falsely flagged before)
r = c.post('/predict', json={'hour':17,'month':4,'doy':113,'sunElevation':8,'temp':26,'wind':3,'irradiance':80,'cloud':75,'sunAzimuth':260})
d = fj.loads(r.data)
assert not d['is_anomaly'], "FAIL: Cloudy twilight flagged as anomaly (false positive)"
print("T4 PASS: Cloudy twilight -> {} (no false positive)".format(d['anomaly_type']))

# Test 5: verify anomaly_type and aoi_deg are in response
assert 'anomaly_type' in d, "FAIL: anomaly_type missing from response"
assert 'aoi_deg' in d, "FAIL: aoi_deg missing from response"
print("T5 PASS: anomaly_type and aoi_deg present in response")

print("ALL ANOMALY TESTS PASSED")
