"""Create degraded test inputs from clean EuRoC sequences, in two modes:
 A = degrade_imu.py applied directly to EuRoC data.csv (gyro-first columns; script assumes accel-first)
 B = columns reordered to accel-first, degraded, then reordered back to EuRoC layout"""
import subprocess, sys, os, json, pandas as pd
E=os.environ.get('EUROC','data/euroc'); O=os.environ.get('DEG','data/deg'); K=os.environ.get('DEGRADE','degrade_imu.py')
for s in sorted(os.listdir(E)):
    for mode in 'AB':
        d=f'{O}/{mode}/{s}'; os.makedirs(d,exist_ok=True)
        src=f'{E}/{s}/imu.csv'
        if mode=='A':
            subprocess.run([sys.executable,K,src,f'{d}/imu.csv','--meta',f'{d}/meta.json'],check=True,capture_output=True)
        else:
            df=pd.read_csv(src); c=list(df.columns)
            df[[c[0]]+c[4:7]+c[1:4]].to_csv(f'{d}/tmp.csv',index=False)
            subprocess.run([sys.executable,K,f'{d}/tmp.csv',f'{d}/tmp2.csv','--meta',f'{d}/meta.json'],check=True,capture_output=True)
            o=pd.read_csv(f'{d}/tmp2.csv'); oc=list(o.columns)
            o=o[[oc[0]]+oc[4:7]+oc[1:4]]
            o.to_csv(f'{d}/imu.csv',index=False); os.remove(f'{d}/tmp.csv'); os.remove(f'{d}/tmp2.csv')
        m=json.load(open(f'{d}/meta.json'))
        print(mode,s,m['rows_in'],m['rows_out'],m['dropped_total'],m['duplicated_rows'],m['spikes'],m['stuck_samples'],m['final_gyro_bias_rad_s'])
