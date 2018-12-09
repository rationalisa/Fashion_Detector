function year = getYear(imname)
    Iinfo = imfinfo(imname);
    year = str2double(Iinfo.Filename(40:43)); 
end