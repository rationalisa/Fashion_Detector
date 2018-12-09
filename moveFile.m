function year = moveFile()
    info  = dir( fullfile('./9697', '*.jpg'));
    filenames = fullfile('./9697', {info.name} );
    len=length(filenames);
    for i=1:len
        if rem(i,5)==0
            year =movefile(filenames{i},'./new');
        end
        
    end
    end