# York scenarios

Location: -31.8912, 116.7683

- PGA at 500 RP: 5.935e-2
- PGA at 1000 RP: 1.0203e-1
- PGA at 2500 RP: 1.9875e-1
#- PGA at 5000 RP: 3.1205e-1

|jobID|version|Note|
|:--:|:--:|:--:|
|1|3.5|classical|
|2|3.3.2|classical|
|3|3.1|classical|
|4|3.5|classical no minimum PGA|
|17|3.5|event based|
|19|3.5|event based|


|Event|Mw|Depth(km)|
|:--:|:--:|:--:|
|Calingiri| 5.03|15| 5-5.2|
|Lake Muir | 5.3 | 2 | 5.2-5.4|
|Cadoux|6.1 | 3 | 6-6.2|
|Meckering| 6.58| 10| 6.4-6.6|


rupid = eid // 2 ** 32

PGA at 500 RP

Calingiri': {1: ('d10', 'lo24', 'la20', 191.0, 14.546839299314547
# lo24: 116.6-116.8
# 1a20: -31.9 -31.7

sites = 116.7683 -31.8912

16457293 

grep -in "16457293" ./ses_11.xml
sed -n '20643352,20644352 p' ./ses_11.xml > new_file

PGA at 1000 RP

'Lake Muir': {1: ('d10', 'lo25', 'la20', 109.0, 
16.795069337442218)
# lo25: 116.8-117.0
# 1a20: -31.9 -31.7

37526917
grep -in "37526917" ./ses_11.xml


PGA at 2500 RP

Meckering': {1: ('d10', 'lo26', 'la19', 19.0, 20.43010752688172)
14046212


# d10: 4.5-5.0
# lo26: 117.0-117.2
# la19: -32.1 -31.9

# lo24: 116.6-116.8
# lo25: 116.8-117.0
# lo26: 117.0-117.2

# la19: -32.1 -31.9
# 1a20: -31.9 -31.7

# running rock scenario hazard
 python -m openquake.commands.__main__ engine --run /c/W10Dev/York/Rp500/job_rock.ini --exports csv

# export realizations_{id}.csv
 python -m openquake.commands.__main__ engine --export-output 88 /c/W10Dev/York/Rp500/output/

# export 
 python -m openquake.commands.__main__ show events > /c/W10Dev/York/Rp500/output/eid2rlz_543.csv

 z1pt0, z2pt5

 In [4]: a['vs30'].value_counts()
Out[4]:
270     2755
412     1176
560     1169
760      525
1100     240
180      168
115       27
Name: vs30, dtype: int64

# In [5]: a['z1pt0'].value_counts()
z1pt0 = {270: 479.445596,
412: 340.738604,
560: 162.230898,
760: 41.306642,
1100: 4.248040,
180: 514.021909,
115: 521.595634}

# In [6]: a['z2pt5'].value_counts()
z2pt5 = {270: 2.240210,
412: 1.742252,
560: 1.101409,
760: 0.667291,
1100: 0.534250,
180: 2.364339,
115: 2.391528}

# three intervals:
- 10, 20, 30
- 15, 30, 45 bldgs (10 bldgs from heritage-listed, 5 from non-heritage)


