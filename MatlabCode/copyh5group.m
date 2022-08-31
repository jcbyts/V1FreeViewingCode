function copyh5group(fOrigin, fTarget, GroupName, append)
% copyh5group(fOrigin, fTarget, GroupName, append)
% copy an hdf5 Group from one file to another. Append a string to the new
% group
% fOrigin - file name of the h5 file to copy FROM
% fTarget - file name of the h5 file to copy TO
% GroupName - the group name to copy
% append - a string to append to the group name in the target file


hinfo = h5info(fOrigin, ['/' GroupName]);

n = numel(hinfo.Groups);
for i = 1:n
    m = numel(hinfo.Groups(i).Datasets);
    GroupNameOrigin = hinfo.Groups(i).Name;
    GroupNameTarget = strrep(GroupNameOrigin, GroupName, [GroupName append]);

    for j = 1:m
        try
            Name = hinfo.Groups(i).Datasets(j).Name;

            h5pathOrigin = [GroupNameOrigin '/' Name];
            h5pathTarget = [GroupNameTarget '/' Name];

            data = h5read(fOrigin, h5pathOrigin);
            sz = size(data);

            fprintf('Reading [%s] from origin. Size: [', h5pathOrigin)
            fprintf('%d ',  sz)
            fprintf(']\n')

            fprintf('Writing [%s] to target\n', h5pathTarget)
            h5create(fTarget,h5pathTarget, sz);
            h5write(fTarget, h5pathTarget, data)

            if ~isempty(hinfo.Groups(i).Datasets(j).Attributes)
                natt = numel(hinfo.Groups(i).Datasets(j).Attributes);
                for a = 1:natt

                    attName = hinfo.Groups(i).Datasets(j).Attributes(a).Name;
                    attVal = hinfo.Groups(i).Datasets(j).Attributes(a).Value;
                    fprintf('writing attribute [%s]\n', attName)
                    h5writeatt(fTarget, h5pathTarget, attName, attVal)
                end

            end
        end
    end
end
