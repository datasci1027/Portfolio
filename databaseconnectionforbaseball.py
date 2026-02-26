# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:06:30 2023

@author: Aaron
"""

import psycopg2
from config import config
import pandas as pd
import sys

def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config()
  
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        
    
        # create a cursor
        cur = conn.cursor()
          
    # execute a statement
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')
  
        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)
         
        #code dealing with the command line
        filename=str(sys.argv[1])
        print("pulling intended file:" + " " + filename)

        
        
        #code for opening the file and pulling the info where needed from the csv into the variables and array as needed.
        with open(filename) as f:
                    next(f)
                    next(f)
                    #pull in name
                    playername = next(f)
                    name=playername.split(",")
                    name=name[1].split(" ")
                    firstname=name[0]
                    lastname=name[1].split("\n")[0]
                    print(firstname)
                    print(lastname)
                    
                    data = pd.read_csv(f)
                    data['firstname']=firstname
                    data['lastname']=lastname
                    print(data.shape)
                    
                    #code for using the array of data, collected from the csv, to insert (copy) data into the database
                    for index, row in data.iterrows():
                        
                        
                        query =  'INSERT INTO rapsodo_pitch_t VALUES %s ON CONFLICT (pitchtime, DeviceSerialNumber) DO NOTHING;'
                        data = (tuple(row),)
                        cur.execute(query, data)
                        
                    
      #other code needed for any errors that occur relating to errors with the database. Also code for telling us the database is closed and that also closes (turns off) the cursor, database connection, and other connections.
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            print('Database connection closed.')
            conn.commit()
            cur.close()
            conn.close()
if __name__ == '__main__':
    connect()