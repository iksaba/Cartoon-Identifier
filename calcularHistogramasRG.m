function histograma2D = calcularHistogramasRG(imagen, numBins)
% Esta función calcula un histograma bidimensional normalizado para r y g
% Parámetros:
%   imagen - Imagen a analizar
%   numBins - Número de divisiones para el histograma
% Retorna:
%   histograma2D - Histograma bidimensional de r-g normalizado

% Verificar si es una imagen en color (RGB)
if size(imagen, 3) == 3
    % Extrae los canales RGB
    canal_R = double(imagen(:,:,1));
    canal_G = double(imagen(:,:,2));
    canal_B = double(imagen(:,:,3));
    
    % Calcular la suma de los tres canales para cada píxel
    suma_RGB = canal_R + canal_G + canal_B;
    
    % Evitar división por cero
    suma_RGB(suma_RGB == 0) = 1;
    
    % Normalizar los valores r y g
    r_norm = canal_R ./ suma_RGB;
    g_norm = canal_G ./ suma_RGB;
    
    % Convertir los valores normalizados a índices para el histograma
    % Los valores estarán entre 0 y 1, los escalamos a 1...numBins
    r_indices = max(1, min(numBins, floor(r_norm * numBins) + 1));
    g_indices = max(1, min(numBins, floor(g_norm * numBins) + 1));
    
    % Inicializar el histograma 2D
    histograma2D = zeros(numBins, numBins);
    
    % Calcular el histograma bidimensional  
    for i = 1:numel(r_indices)
        histograma2D(g_indices(i), r_indices(i)) = histograma2D(g_indices(i), r_indices(i)) + 1;
    end
    
    % Normalizar el histograma para que la suma sea 1
    histograma2D = histograma2D / sum(histograma2D(:));
else
    error('La imagen debe ser RGB (3 canales)');
end