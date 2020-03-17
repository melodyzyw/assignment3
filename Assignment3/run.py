import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import matplotlib
import json
import os
from datetime import *

from sklearn.linear_model import LogisticRegression

import run
warnings.filterwarnings("ignore")



def main(targets):

    if 'data-test' in targets:
        with open("config/test-project.json") as fh:
            data_cfg = json.load(fh)
        run.data_test(**data_cfg)
    return

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)

def data_test(file_path, df_path, write_path):
    find_percentage(file_path, clean_df(df_path), write_path)


def find_percentage(file_path, df, write_path):
    t_17 = pd.read_csv(file_path).loc[:364]
    t_18 = pd.read_csv(file_path).loc[365:]
    t_17['Day'] = t_17['Day'].apply(lambda x:x+' 2017')
    t_18['Day'] = t_18['Day'].apply(lambda x:x+' 2018')
    t_17['Sunset'] = t_17['Day'] + ' ' + t_17['Sunset']
    t_17['Dusk'] = t_17['Day'] + ' ' + t_17['End']
    t_18['Sunset'] = t_18['Day'] + ' ' + t_18['Sunset']
    t_18['Dusk'] = t_18['Day']+' '+t_18['End']
    t_17.drop(columns = ['Day', 'End'], inplace=True)
    t_18.drop(columns = ['Day', 'End'], inplace=True)
    t_17 = t_17.applymap(lambda x:datetime.strptime(x, '%a, %b %d %Y %I:%M:%S %p'))
    t_18 = t_18.applymap(lambda x:datetime.strptime(x, '%a, %b %d %Y %I:%M:%S %p'))
    twilight = t_17.append(t_18)
    twilight['date_stop'] = twilight['Sunset'].apply(lambda x:datetime.date(x)).astype(str)
    vod = df.merge(twilight, on='date_stop')
    #datetime to string (take 'time' only)
    vod.Sunset = vod.Sunset.apply(lambda x: datetime.strftime(x, '%H:%M:%S'))
    vod.Dusk = vod.Dusk.apply(lambda x: datetime.strftime(x, '%H:%M:%S'))
    vod['stop'] = vod.timestamp.apply(lambda x: datetime.strftime(x, '%H:%M:%S'))
    #tring to datatime (make all the years the same to compare time)
    vod.Sunset = vod.Sunset.apply(lambda x: datetime.strptime(x, '%H:%M:%S'))
    vod.Dusk = vod.Dusk.apply(lambda x: datetime.strptime(x, '%H:%M:%S'))
    vod.stop = vod.stop.apply(lambda x: datetime.strptime(x, '%H:%M:%S'))
    #chosen inter-twilight period
    set_start = datetime(1900, 1, 1, 18, 45, 0)
    set_stop = datetime(1900, 1, 1, 19, 0, 0)
    #Stop time is between the chosen period
    vod = vod[vod.stop >= set_start]
    vod = vod[vod.stop <= set_stop]

    #Stop time is NOT between the inter-twilight period of that day (should be neither dark or bright)
    bright = vod[list(vod.stop <= vod.Sunset)]
    dark = vod[list(vod.stop >= vod.Dusk)]

    #Whether the stop time is before sunset or not (bright or not)
    #vod['Bright'] = list(vod.stop <= vod.Sunset)
    temp1 = dark[['stop_id','subject_race']]
    temp1['Dark'] = [True for x in temp1['stop_id']]

    temp2 = bright[['stop_id','subject_race']]
    temp2['Dark'] = [False for x in temp2['stop_id']]

    regres = temp1.append(temp2)
    regres['Black'] = [True if x == 'Black/African American' else False for x in regres['subject_race']]

    x=regres.Dark.values.reshape(-1,1)

    y=regres.Black

    clf = LogisticRegression(random_state=0).fit(x, y)

    dic = {"clf.coef":str(clf.coef_)}

    write_to_disk(write_path, dic, "output.json")


def write_to_disk(write_path, dictionary, name):
    with open(os.path.join(write_path, name), 'w') as fp:
        json.dump(dictionary, fp)

def clean_df(df_path):
    df = pd.read_csv(df_path)
    #df = pd.concat([df14, df15, df16, df17])

    #clean race
    race_code = {'A':'Asian', 'B':'Black/African American', 'C':'Asian', 'D':'Asian', 'F':'Asian', 'G':'Pacific Islander', 'H':'Hispanic/Latino/a', 'I':'Middle Eastern or South Asian', 'J':'Asian', 'K':'Asian', 'L':'Asian', 'O':np.nan, 'P':'Pacific Islander', 'S':'Pacific Islander', 'U':'Pacific Islander', 'V':'Asian', 'X':np.nan, 'Z':'Middle Eastern or South Asian', 'W':'White'}
    df = df.replace({'subject_race': race_code})
    df = df.dropna(subset=['subject_race', 'date_time'])

    #service area
    area_code = {'110':'Northern', '120':'Northern', '130':'Northern',
                '230':'Northeastern', '240':'Northeastern',
                '310':'Eastern', '320':'Eastern',
                '430':'Southeastern', '440':'Southeastern',
                '510':'Central', '520':'Central', '530':'Central',
                '610':'Western', '620':'Western', '630':'Western',
                '710':'Southern', '720':'Southern',
                '810':'Mid-City', '820':'Mid-City', '830':'Mid-City', '840':'Mid-City',
                '930':'Northwestern',
                'Unknown':np.nan}
    df['division'] = df['service_area']
    df = df.replace({'division': area_code})

    #age
    df['subject_age'] = pd.to_numeric(df['subject_age'], errors='coerce')
    df['subject_age'] = df.subject_age.apply(lambda x: np.NaN if ((float(x) < 14) | (float(x) > 99)) else x)

    #date time
    #df['timestamp'] = pd.to_datetime(df['date_time'], errors='coerce')
    df['timestamp'] = df['date_time'].dropna().apply(lambda x:datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

    #searched
    #resident
    df['searched'] = df.searched.apply(clean_binary_cols)
    df['sd_resident'] = df.sd_resident.apply(clean_binary_cols)

    return df

def impute_target(stops):
    def impute(searched):
        p = searched.mean()
        return searched.apply(
            lambda x: np.random.choice([0, 1], p=[1 - p, p]) if pd.isnull(x) else x
        )
    searched = (
        stops
        .fillna({'service_area': 'NULL'})
        .groupby('service_area')
        .searched
        .apply(impute)
    )
    return stops.assign(searched=searched)

def clean_binary_cols(x):
    if x in ['Y', 'y']:
        return 1
    elif x in ['N', 'n']:
        return 0
    else:
        return np.NaN

def get_clean_table_old(year):
	url='http://seshat.datasd.org/pd/vehicle_stops_{}_datasd_v1.csv'.format(str(year))
	df = pd.read_csv(url)

 	#clean race
	race_code = {'A':'Asian', 'B':'Black/African American', 'C':'Asian', 'D':'Asian', 'F':'Asian', 'G':'Pacific Islander', 'H':'Hispanic/Latino/a', 'I':'Middle Eastern or South Asian', 'J':'Asian', 'K':'Asian', 'L':'Asian', 'O':np.nan, 'P':'Pacific Islander', 'S':'Pacific Islander', 'U':'Pacific Islander', 'V':'Asian', 'X':np.nan, 'Z':'Middle Eastern or South Asian', 'W':'White'}
	df = df.replace({'subject_race': race_code})
	df = df.dropna(subset=['subject_race'])

	#service area
	area_code = {'110':'Northern', '120':'Northern', '130':'Northern',
				'230':'Northeastern', '240':'Northeastern',
				'310':'Eastern', '320':'Eastern',
				'430':'Southeastern', '440':'Southeastern',
				'510':'Central', '520':'Central', '530':'Central',
				'610':'Western', '620':'Western', '630':'Western',
				'710':'Southern', '720':'Southern',
				'810':'Mid-City', '820':'Mid-City', '830':'Mid-City', '840':'Mid-City',
				'930':'Northwestern',
				'Unknown':np.nan}
	df['division'] = df['service_area']
	df = df.replace({'division': area_code})

	#age
	df['subject_age'] = pd.to_numeric(df['subject_age'], errors='coerce')
	df['subject_age'] = df.subject_age.apply(lambda x: np.NaN if ((float(x) < 14) | (float(x) > 99)) else x)

	#date time
	df['timestamp'] = pd.to_datetime(df['date_time'], errors='coerce')

	#searched
	#resident
	df['searched'] = df.searched.apply(clean_binary_cols)
	df['sd_resident'] = df.sd_resident.apply(clean_binary_cols)

	return df


def get_clean_table_ripa(year):
	df19 = pd.read_csv('http://seshat.datasd.org/pd/ripa_stops_datasd_v1.csv')
	features = ['actions_taken', 'prop_seize_basis', 'race', 'search_basis', 'stop_reason', 'stop_result']
	for i in features:
		table = pd.read_csv(('http://seshat.datasd.org/pd/ripa_{}_datasd.csv').format(i))
		df19 = df19.merge(table, on=['stop_id', 'pid'])
		df19 = df19[['stop_id', 'reason_for_stop', 'beat', 'race', 'gend', 'perceived_age', 'date_stop', 'time_stop', 'action', 'result', 'basisforpropertyseizure']]
	return df19


def get_data(years, outpath):
	if os.path.exists(outpath) == False:
		os.mkdir(outpath)
	for i in years:
		if int(i) < 2018:
			table = get_clean_table_old(i)
			out = '%s/%s_stops.csv'%(str(outpath), str(i))
			table.to_csv(out)
		else:
			table = get_clean_table_ripa(i)
			out =' %s/%s_stops.csv'%(str(outpath), str(i))
			table.to_csv(out)
