function gaussianBlur=gaussianBlur(image, sigma, kernelSize)


    kernelSize = round(sigma*3 - 1); 
    if(kernelSize<1)
        kernelSize = 1; 
    end 
	kernel = fspecial('gaussian', [kernelSize kernelSize], sigma);

	convImage = imfilter(image,kernel,'replicate');
%	convImage = imfilter(image,kernel);

	gaussianBlur = convImage; 
end
