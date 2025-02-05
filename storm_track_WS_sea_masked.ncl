load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRF_contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"

begin
  ; Path to WRF output files
  Path = "/shared/airmg/ISRAT/ML_WG/WRF_Extraction_Done/rainwind_anl_WRF_4.2/from_temp_3/2021042112/wrfout_files_d02/"
  
  ; List all WRF files
  NC_Files = systemfunc("ls "+Path+"wrfout_d02*")
  hours = dimsizes(NC_Files)

  do i=0,hours-1
     filename = NC_Files(i)
     a = addfile(filename,"r")

     ; Read variables
     lat2d = a->XLAT(0,:,:)
     lon2d = a->XLONG(0,:,:)
     landmask = a->LANDMASK(0,:,:)  ; 1 for land, 0 for ocean

     ; Read U10 and V10 for wind components
     u = wrf_user_getvar(a,"U10",0)
     v = wrf_user_getvar(a,"V10",0)

     ; Compute wind speed and apply landmask
     WS_2d = (u^2 + v^2)^0.5
     WS_2d = WS_2d * landmask  ; Set wind speed to 0 over ocean

     ; Flatten arrays for indexing
     WS_1d = ndtooned(WS_2d)
     lon1d = ndtooned(lon2d)
     lat1d = ndtooned(lat2d)

     ; Find maximum wind speed on land
     WS_max = max(WS_1d)
     max_idx = maxind(WS_1d)

     ; Get corresponding latitude and longitude
     max_lat = lat1d(max_idx)
     max_lon = lon1d(max_idx)

     ; Print results
     lines = "Hour,"+i+","+sprintf("%14.6f",max_lon)+","+sprintf("%14.6f",max_lat)+","+sprintf("%5.2f",WS_max)
     print(lines)
     asciiwrite("comp"+i+".csv",lines)
  end do

  ; Concatenate results into a single file
  ConcentrateFiles = systemfunc("cat comp*.csv > max_WS_land_2021042112.csv")
  MoveAllCompiles = systemfunc("rm comp*.csv")
end

