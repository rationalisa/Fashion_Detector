function safe_rmdir(fname)

try
  rmdir(fname);
catch 
end