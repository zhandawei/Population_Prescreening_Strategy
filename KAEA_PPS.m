% A population precreening strategy for Kriging-assisted evolutionary
% algorithms. The proposed algorithm generates multiple candidate
% populations in each generation and uses the qEI to prescreening these
% candidates for selecting the best population to work with.
% Dawei Zhan, Huanlai Xing, A population prescreening strategy for
% Kriging-assisted evolutioanry computation, IEEE Congress on Evolutionary
% Computation, 2021.doi: 10.1109/CEC45853.2021.9504976.
% -------------------------------------------------------------------------
clearvars;clc;close all;
fun_name = 'Ellipsoid';
num_vari = 10;
% population size
pop_size = 30;
% number of expensive evaluations in each generation (num_q<=pop_size)
num_q = 5;
% number of candidate populations
tau = 20;
% maximum number of expensive evaluations
max_evaluation = 500;
% number of initial samples
num_initial = 150;
% parameters of the DE algorithm
CR = 0.8;
F = 0.8;
% lower and upper bounds of the population
lower_bound = -5.12*ones(1,num_vari);
upper_bound = 5.12*ones(1,num_vari);
% number of current generation
generation = 1;
% generate random samples
sample_x = lhsdesign(num_initial, num_vari,'criterion','maximin','iterations',1000).*(upper_bound - lower_bound) + lower_bound;
sample_y = feval(fun_name, sample_x);
evaluation =  size(sample_x,1);
% best objectives in each generation
[fmin,idx] = min(sample_y);
xmin = sample_x(idx,:);
% print the iteration information
fprintf('KAEA-PPS on %s%d, generation: %d, evaluation: %d, best: %0.4g\n',fun_name,num_vari,generation,evaluation,fmin);
% the first generation of DE
[~,index] = sort(sample_y);
pop_vari = sample_x(index(1:pop_size),:);
pop_obj = sample_y(index(1:pop_size),:);
while evaluation < max_evaluation
    % build the Kriging model
    kriging_model = dacefit(sample_x,sample_y,'regpoly0','corrgauss',1*ones(1,num_vari),0.001*ones(1,num_vari),1000*ones(1,num_vari));
    qEI = zeros(tau,1);
    % record candidate populations
    pop_trial_record = zeros(pop_size,num_vari,tau);
    for gen = 1 : tau
        % mutation
        pop_mutation = zeros(pop_size,num_vari);
        for ii = 1 : pop_size
            avail_num = (1:pop_size);
            avail_num(ii) = [];
            % randomly generate three different integers
            r = avail_num(randperm(length(avail_num),2));
            pop_mutation(ii,:) = xmin + F*(pop_vari(r(1),:)-pop_vari(r(2),:));
            % check the bound constraints, randomly re-initialization
            if any(pop_mutation(ii,:)<lower_bound) || any(pop_mutation(ii,:)>upper_bound)
                pop_mutation(ii,:) = lower_bound + rand(1,num_vari).*(upper_bound-lower_bound);
            end
        end
        % crossover
        rand_matrix = rand(pop_size,num_vari);
        temp = randi(num_vari,pop_size,1);
        for ii = 1 : pop_size
            rand_matrix(ii,temp(ii)) = 0;
        end
        mui = rand_matrix < CR;
        pop_trial = pop_mutation.*mui + pop_vari.*(1-mui);
        qEI(gen) = qEI_MC(pop_trial,kriging_model,fmin,100000);
        pop_trial_record(:,:,gen) = pop_trial;
    end
    % select the population from tau candidates
    [max_qEI,max_index] = max(qEI);
    % select infill samples
    pop_trial = pop_trial_record(:,:,max_index);
    % remove reduplicative points
    pop_candi = unique(pop_trial(~ismember(pop_trial,sample_x,'rows'),:),'rows');
    [u,s] = predictor(pop_candi,kriging_model);
    % calculate EI values
    EI = (fmin-u).*normcdf((fmin-u)./s)+s.*normpdf((fmin-u)./s);
    [sort_EI,ind] = sort(EI,'descend');
    infill_x = pop_candi(ind(1:num_q),:);
    % expensive evaluations
    infill_y = feval(fun_name, infill_x);
    % replacement
    [~,index] = ismember(infill_x,pop_trial,'rows');
    replace_index = index(infill_y <= pop_obj(index));
    pop_vari(replace_index,:) = pop_trial(replace_index,:);
    pop_obj(replace_index,:) = infill_y(infill_y <= pop_obj(index));
    % update database
    sample_x = [sample_x;infill_x];
    sample_y = [sample_y;infill_y];
    % update the evaluation number
    generation = generation + 1;
    evaluation = evaluation + size(infill_x,1);
    [fmin,idx] = min(sample_y);
    xmin = sample_x(idx,:);
    % print the iteration information
    fprintf('KAEA-PPS on %s%d, generation: %d, evaluation: %d, best: %0.4g\n',fun_name,num_vari,generation,evaluation,fmin)
end







