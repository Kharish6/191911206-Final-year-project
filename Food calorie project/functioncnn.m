function rnn=functioncnn(rnn, xx)

if rnn.no_of_input_channels > 1
    for i=1:rnn.no_of_input_channels 
        rnn.layers{1}.featuremaps{i}=xx(:,:,i,:);
    end
else
    rnn.layers{1}.featuremaps{1}=xx;
end
for i=2:rnn.no_of_layers
    
    if rnn.layers{i}.type == 'c'
        kk=0;
        zz=0;
        for j=1:rnn.layers{i}.no_featuremaps
            z = 0; 
            for k=1:rnn.layers{i-1}.no_featuremaps
                kk = kk +1;
                z = z + convn(rnn.layers{i-1}.featuremaps{k},rot90(rnn.layers{i}.K(:,:,kk),2),'valid'); 

            end
            if rnn.layers{i}.act_func == 'soft'
                rnn.layers{i}.featuremaps{j}= exp(z + rnn.layers{i}.b(j));
                zz = zz + rnn.layers{i}.featuremaps{j};
            else
                rnn.layers{i}.featuremaps{j} = applyactivationfunccnn(z+ rnn.layers{i}.b(j),rnn.layers{i}.act_func, 0);
%                 
            end
        end
        if rnn.layers{i}.act_func == 'soft'
            for j=1:rnn.layers{i}.no_featuremaps
                rnn.layers{i}.featuremaps{j}= rnn.layers{i}.featuremaps{j} ./ zz;
            end
        end
    elseif rnn.layers{i}.type == 'p'
        
            if rnn.layers{i}.subsample_method == 'mean'
                h = ones([rnn.layers{i}.subsample_rate rnn.layers{i}.subsample_rate]); h=h./sum(h(:));
                for k=1:rnn.layers{i-1}.no_featuremaps
                    zz = convn(rnn.layers{i-1}.featuremaps{k}, h, 'valid'); %%'same'
                    rnn.layers{i}.featuremaps{k} = zz(1:rnn.layers{i}.subsample_rate:end, 1:rnn.layers{i}.subsample_rate:end,:);

                end
            elseif rnn.layers{i}.subsample_method == 'max '
                error 'max pooling not implemented'

            end
    elseif rnn.layers{i}.type == 'f'
            zz=0;
            zz=[];
            if rnn.layers{i-1}.type  ~= 'f'
                for k=1:rnn.layers{i-1}.no_featuremaps
                   ss =size(rnn.layers{i-1}.featuremaps{k});
                   ss(3) =size(rnn.layers{i-1}.featuremaps{k},3);
                   if rnn.input_image_width == 1
                       ss(3) =ss(2);
                       ss(2)=1;
                   end
                   zz =[zz; reshape(rnn.layers{i-1}.featuremaps{k}, ss(1)*ss(2), ss(3))];
                   
                end
                rnn.layers{i-1}.outputs = zz;
                rnn.layers{i}.outputs = applyactivationfunccnn(rnn.layers{i}.W*zz + repmat(rnn.layers{i}.b, 1, size(zz,2)), rnn.layers{i}.act_func, 0); 

         
            else
                zz= rnn.layers{i-1}.outputs;
                rnn.layers{i}.outputs = applyactivationfunccnn(rnn.layers{i}.W*zz + repmat(rnn.layers{i}.b, 1, size(zz,2)), rnn.layers{i}.act_func, 0); 
            end
                
        
    end
    
end