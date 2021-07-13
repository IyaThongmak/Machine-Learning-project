#plotgraph
        
    latitude = float(Latitude.get()) 
    longtitude = float(Longtitude.get())


    # Create the map plotter:
    
    apikey = 'AIzaSyBDfgSoym1wzkbN1tXJ8cIjs4pxoo4AKd4' # (your API key here)
    gmap = gmplot.GoogleMapPlotter(latitude,longtitude, 14, apikey=apikey)
  
    # Mark a hidden gem:
    gmap.marker(latitude,longtitude, color=color)
    #gmap.marker(float(Latitude2.get()),float(Longtitude2.get()), color=color)
    #gmap.marker(float(Latitude3.get()) ,float(Longtitude3.get()), color=color)


    print ('start drawing')
    
    # Draw the map:
    gmap.draw('map7.html')
    print('success')
    #End plotgraph
