# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 12:33:30 2018

@author: Connor Higgins
"""

def RouteOptimizer(origins, destinations):

    ## SETUP ##
    from googlemaps.client import Client
    from googlemaps.distance_matrix import distance_matrix
    import numpy as np
    import re
    
    # old api_key = 'AIzaSyAnWAegsWEiyeIWuTk2BIqpIHRoU9MTcQ8'
    api_key = 'AIzaSyBpMVkWFvoy0_awNsCJlTaEI1UA-7wwYlk'
    gmaps = Client(api_key)
    
    ## Origin to Destinations ##
    dataorigin = distance_matrix(gmaps, origins, destinations)
    numdest = len(destinations)
    numorig = len(origins)
    
    # pulling different route times
    routetime = []
    for j in range(numorig):
        for i in range(numdest):
            str1 = dataorigin['rows'][j]['elements'][i]['duration']['text']
            int1 = int(''.join(filter(str.isdigit, str1)))
            routetime.append(int1)
            
    # pulling the shortest duration route  
    first_route_time = min(routetime)
    first_route_index = routetime.index(first_route_time)
    ri1 = first_route_index
    
    # pulling the distance and address of shortest duration route
    first_route_address = dataorigin['destination_addresses'][ri1]
    diststr1 = dataorigin['rows'][0]['elements'][ri1]['distance']['text']
    first_route_distance = float(re.findall(r"[-+]?\d*\.\d+|\d+", diststr1)[0])
    ra1 = first_route_address 
    
    ## Creating a new destinations matrix ##
    from collections import Iterable
    new_destinations = [destinations[ri1], destinations[0:ri1], destinations[ri1+1:]]
    
    # removing nested lists
    def flatten(items):
        """Yield items from any nested iterable; see REF."""
        for x in items:
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                yield from flatten(x)
            else:
                yield x
    new_dest = list(flatten(new_destinations))
    
    ## Destinations to Destinations ##
    datadest = distance_matrix(gmaps, new_dest, new_dest)
    
    # pulling route times for each destination pair
    routetimes = []
    for j in range(numdest):
        for i in range(numdest):
            str1 = datadest['rows'][j]['elements'][i]['duration']['text']
            int1 = int(''.join(filter(str.isdigit, str1)))
            routetimes.append(int1)
            
    # rearranging list into matrix and changing same location time to 999 (so it isn't considered)
    times_organized = np.reshape(routetimes, (numdest, numdest))
    times_organized[times_organized == 1] = 999
    
    # finding all possible routes
    numnewdest = list(np.arange(1, numdest, 1))
    import itertools
    total_paths = list(itertools.permutations(numnewdest))
    
    # finding times for all possible routes
    totaltimes = []
    for sublist in total_paths:
        total = []
        for i in sublist:
            if sublist.index(i) == 0:
                total.append(times_organized[0, i])
            else:
                old_index = sublist.index(i) - 1
                total.append(times_organized[sublist[old_index], i])
        totaltimes.append(np.array(total).sum())
    
    # pulling the shortest duration route time and path
    min_route_time = min(totaltimes)
    min_route_index = totaltimes.index(min_route_time)
    min_route_path = total_paths[min_route_index]
    
    ## Organizing Min_Route Info ##
    min_route_list = list(flatten(min_route_path))
    
    ra0 = dataorigin['origin_addresses'][0]
    min_route_addresses = [ra0, ra1]
    
    # creating a list of addresses in proper route order
    for i in min_route_list:
        ra = datadest['destination_addresses'][i]
        min_route_addresses.append(ra)
        
    return(min_route_addresses, 'total time: {}mins'.format(min_route_time))
