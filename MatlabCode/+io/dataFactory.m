function varargout = dataFactory(varargin)
% DATAFACTORY is a big switch statement. You pass in the ID of the sesion
% you want and it returns the processed data.
%
% Input:
%   Session [Number or ID string]
% Output: 
%   Exp [struct]
%
% Example:
%   Exp = io.dataFactory(5); % load the 5th session

switch nargout
    case 0
        io.dataFactoryGratingSubspace(varargin{:});
    case 1
        varargout{1} = io.dataFactoryGratingSubspace(varargin{:});
    case 2
        [varargout{1},varargout{2}] = io.dataFactoryGratingSubspace(varargin{:});
    case 3
        [varargout{1},varargout{2},varargout{3}] = io.dataFactoryGratingSubspace(varargin{:});
    case 4
        [varargout{1},varargout{2},varargout{3},varargout{4}] = io.dataFactoryGratingSubspace(varargin{:});
    otherwise
        error('dataFactory: unknown number of arguments')
end