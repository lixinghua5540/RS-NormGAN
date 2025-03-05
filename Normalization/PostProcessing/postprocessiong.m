%test area of SHCD
weightMat=zeros(128,128);
for i=1:128
    for j=1:128
        d1=(i-1)*(i-1)+(j-1)*(j-1);
        Dw1=1/(d1+1);
        d2=(i-1)*(i-1)+(j-128)*(j-128);
        Dw2=1/(d2+1);
        d3=(i-128)*(i-128)+(j-1)*(j-1);
        Dw3=1/(d3+1);
        d4=(i-128)*(i-128)+(j-128)*(j-128);
        Dw4=1/(d4+1);
        weightMat(i,j)=Dw1/(Dw1+Dw2+Dw3+Dw4);
    end
end
weightMatD=imrotate(weightMat,90);
weightMatRD=imrotate(weightMatD,90);
weightMatR=imrotate(weightMatRD,90);
for i=0:13
    for j=0:11
        if(((i+2)*200<=10980)&&((j+2)*200<=10980))
            simg=imread(strcat("Path_to_your_result_folder\test1_",num2str(i),"_",num2str(j),"_fake.tif"));
            simgR=imread(strcat("Path_to_your_result_folder\test1_",num2str(i),"_",num2str(j+1),"_fake.tif"));
            simgD=imread(strcat("Path_to_your_result_folder\test1_",num2str(i+1),"_",num2str(j),"_fake.tif"));
            simgRD=imread(strcat("Path_to_your_result_folder\test1_",num2str(i+1),"_",num2str(j+1),"_fake.tif"));
            

            simgr=simg(129:256,129:256,1);
            simgg=simg(129:256,129:256,2);
            simgb=simg(129:256,129:256,3);
            simgr=double(simgr).*weightMat;
            simgg=double(simgg).*weightMat;
            simgb=double(simgb).*weightMat;

            simgRr=simgR(129:256,1:128,1);
            simgRg=simgR(129:256,1:128,2);
            simgRb=simgR(129:256,1:128,3);
            simgRr=double(simgRr).*weightMatR;
            simgRg=double(simgRg).*weightMatR;
            simgRb=double(simgRb).*weightMatR;

            simgDr=simgD(1:128,129:256,1);
            simgDg=simgD(1:128,129:256,2);
            simgDb=simgD(1:128,129:256,3);
            simgDr=double(simgDr).*weightMatD;
            simgDg=double(simgDg).*weightMatD;
            simgDb=double(simgDb).*weightMatD;

            simgRDr=simgRD(1:128,1:128,1);
            simgRDg=simgRD(1:128,1:128,2);
            simgRDb=simgRD(1:128,1:128,3);
            simgRDr=double(simgRDr).*weightMatRD;
            simgRDg=double(simgRDg).*weightMatRD;
            simgRDb=double(simgRDb).*weightMatRD;
            imgr((129+i*128):(256+i*128),(129+j*128):(256+j*128))=uint16(simgr+simgRr+simgDr+simgRDr);
            imgg((129+i*128):(256+i*128),(129+j*128):(256+j*128))=uint16(simgg+simgRg+simgDg+simgRDg);
            imgb((129+i*128):(256+i*128),(129+j*128):(256+j*128))=uint16(simgb+simgRb+simgDb+simgRDb);
        end
    end
end

img(:,:,1)=uint16(imgr);
img(:,:,2)=uint16(imgg);
img(:,:,3)=uint16(imgb);
imgp=img((129+0*128):(256+13*128),(129+0*128):(256+11*128),:);
imwrite(imgp,"Path_to_your_result_folder\Sentinel_test.tif");
%visualization of 16bit image
img=imread("Path_to_your_result_folder\Sentinel_test.tif");
img8bit=((double(img)-700)/1800*255);
imwrite(uint8(img8bit),"Path_to_your_result_folder\Sentinel_testviz.tif");


%testarea of GESD
weightMat=zeros(128,128);
for i=1:128
    for j=1:128
        d1=(i-1)*(i-1)+(j-1)*(j-1);
        Dw1=1/(d1+1);
        d2=(i-1)*(i-1)+(j-128)*(j-128);
        Dw2=1/(d2+1);
        d3=(i-128)*(i-128)+(j-1)*(j-1);
        Dw3=1/(d3+1);
        d4=(i-128)*(i-128)+(j-128)*(j-128);
        Dw4=1/(d4+1);
        weightMat(i,j)=Dw1/(Dw1+Dw2+Dw3+Dw4);
    end
end
weightMatD=imrotate(weightMat,90);
weightMatRD=imrotate(weightMatD,90);
weightMatR=imrotate(weightMatRD,90);
for i=0:5
    for j=0:11
        if(((i+2)*200<=10980)&&((j+2)*200<=10980))
            simg=imread(strcat("Path_to_your_result_folder\A_",num2str(i),"_",num2str(j),"_fake.tif"));
            simgR=imread(strcat("Path_to_your_result_folder\A_",num2str(i),"_",num2str(j+1),"_fake.tif"));
            simgD=imread(strcat("Path_to_your_result_folder\A_",num2str(i+1),"_",num2str(j),"_fake.tif"));
            simgRD=imread(strcat("Path_to_your_result_folder\A_",num2str(i+1),"_",num2str(j+1),"_fake.tif"));
            simgr=simg(129:256,129:256,1);
            simgg=simg(129:256,129:256,2);
            simgb=simg(129:256,129:256,3);
            simgr=double(simgr).*weightMat;
            simgg=double(simgg).*weightMat;
            simgb=double(simgb).*weightMat;

            simgRr=simgR(129:256,1:128,1);
            simgRg=simgR(129:256,1:128,2);
            simgRb=simgR(129:256,1:128,3);
            simgRr=double(simgRr).*weightMatR;
            simgRg=double(simgRg).*weightMatR;
            simgRb=double(simgRb).*weightMatR;

            simgDr=simgD(1:128,129:256,1);
            simgDg=simgD(1:128,129:256,2);
            simgDb=simgD(1:128,129:256,3);
            simgDr=double(simgDr).*weightMatD;
            simgDg=double(simgDg).*weightMatD;
            simgDb=double(simgDb).*weightMatD;

            simgRDr=simgRD(1:128,1:128,1);
            simgRDg=simgRD(1:128,1:128,2);
            simgRDb=simgRD(1:128,1:128,3);
            simgRDr=double(simgRDr).*weightMatRD;
            simgRDg=double(simgRDg).*weightMatRD;
            simgRDb=double(simgRDb).*weightMatRD;

            imgr((129+i*128):(256+i*128),(129+j*128):(256+j*128))=uint8(simgr+simgRr+simgDr+simgRDr);
            imgg((129+i*128):(256+i*128),(129+j*128):(256+j*128))=uint8(simgg+simgRg+simgDg+simgRDg);
            imgb((129+i*128):(256+i*128),(129+j*128):(256+j*128))=uint8(simgb+simgRb+simgDb+simgRDb);
        end
    end
end

img(:,:,1)=uint8(imgr);
img(:,:,2)=uint8(imgg);
img(:,:,3)=uint8(imgb);
imgp=img((129+0*128):(256+5*128),(129+0*128):(256+11*128),:);
imwrite(imgp,"Path_to_your_result_folder\GESD.tif");

