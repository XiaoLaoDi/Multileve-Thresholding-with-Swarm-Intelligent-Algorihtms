function  CSO_result=CSO_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
% function CSO_result=CSO_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
%   The CSO(Cat Swarm Optimization),proposed by Shu-Chuan Chu,Pei-wei Tsai,and Jeng-Shyang Pan in 2006,is try to minimize some optimization problems
%   calculating the maximum Kapur-Entropy/Otsu value use the Cat Swarm Optimization
% Input:
%    FunctionName:the optimized method---Kapur_Entropy/Otsu
%    TH_num:number of thresholds
%    RunTimes:the Repeat Algorithm Running Times
%    MAX_Iterations:
% Output:
%    CSO_result: is a 'Struct' containing the following result information
%        CSO_result.Fitness:
%            CSO_result.Fitness.Mean
%            CSO_result.Fitness.Variance
%            CSO_result.Fitness.Max
%            CSO_result.Fitness.Min
%        CSO_result.Success_Rate=sum(Success_FindBest_Num)/RunTimes;
%        CSO_result.BestThresholds: a vector containing 'TH_num' values
%        CSO_result.EachRunConvergenceTime:a vector containing 'RunTimes' values
%        CSO_result.MeanConvergenceTime: the mean convergence time

disp('the Cat Swarm Optimization is running...')

%% Cat Swarm Optimization STEPS
    %% STEP ONE
        % Initialize the CSO parameters
            global LP nd st BEST_EXHAUSTIVE_FITNESS EPS Gray_image TH_Char ImageName Alg_Name;
            fitness=FunctionName;
            T=MAX_Iterations;                                                  % Maximum Iterate numbers
            N=40;                                                   % number of cats
            SMP=3;                                                  % Seeking Memory Pool
            c=2;                                                    % the coefficient of tracing mode
            SPC=0;                                                  % Self-Position Considering
            SRD=0.10;                                               % Seeking Range of the selected Dimension
            D=TH_num;                                               % the number of thresholds
            CDC=round(0.4*D);                                       % Counts of Dimensions to change
            MR=0.20;                                                % Mixture Ratio

%% 预分配内存
CSO_result.EachRunBestThresholds=zeros(RunTimes,TH_num);
CSO_result.EachRunBestFitness=zeros(1,RunTimes);
CSO_result.BestThresholds=zeros(1,TH_num);
CSO_result.EachRunConvergenceTime=zeros(1,RunTimes);
CSO_result.EachRunConvergenceFunCalNum=zeros(1,RunTimes);
CSO_result.EachRunEveryIterBestFitness=zeros(RunTimes,T);
CSO_result.EachRunEveryIterConvergenceTime=zeros(RunTimes,T);

%% 最外层循环，让CSO跑RunTimes次
Success_FindBest_Index=zeros(1,RunTimes);

for RepeatTimes=1:RunTimes
    rng(sum(RepeatTimes*nd*3000), 'twister');  
    
    tic
    FunCount=0;
    EachRunFunCalNum=zeros(1,T);
    
    %% STEP TWO
        % Random initialize the Cat Swarm Optimization Cat's position and velocity
            x=zeros(N,D);
            for i=1:N
                for j=1:D
                    x(i,j)=(j-1)*floor(nd/D)+floor(rand*floor(nd/D));       % randomized position                                        
                end
            end
            v=rand(N,D);                                                    % randomized velocity
        % Calculate the fitness of each cat 
            Cat_fitness=zeros(1,N);
            for i=1:N    
                Cat_fitness(i)=1/fitness(LP,x(i,:));
                FunCount=FunCount+1;
            end
            
    %% STEP THREE
        %% the CSO main iterations
            Each_Iterate_BestFitness=zeros(1,T);                    % the optimal fitness of each iteration
            Each_Iterate_BestThresholds=zeros(T,D);                 % the optimal solution of each iteration
            Iterates=1;
            Success_FindBest_Index(RepeatTimes)=0;
             while(~Success_FindBest_Index(RepeatTimes) && Iterates<=T) 
%                 CDC=round(((T-t+1)/T)*D); % the results do not improve with this strategy in the Low Dimensional Image
                                            % Thresholding, but it proved working in high dimensional of continue problem              
                % Find the best cat
                    [Each_Iterate_BestFitness(Iterates),index]=min(Cat_fitness);
                    Each_Iterate_BestThresholds(Iterates,:)=x(index,:);
                    if (abs(fitness(LP,Each_Iterate_BestThresholds(Iterates,:))-BEST_EXHAUSTIVE_FITNESS)<=EPS)
                        CSO_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
                        EachRunFunCalNum(Iterates)=FunCount;
                        Success_FindBest_Index(RepeatTimes)=1;
                        break;
                    end                    
                % Haphazardly pick number of cats and set them into tracing mode according to MR, and the others set into seeking mode
                    Tracing_Flag=zeros(1,N);
                    for i=1:N
                        if rand < MR
                            Tracing_Flag(i)=1;
                        else
                            Tracing_Flag(i)=0;
                        end
                    end
                % Move the cats according to their flags, if cat_k is in seeking mode,apply the cat to the seeking mode
                % process,otherwise apply it to the tracing mode process.
                    x_trace=zeros(sum(Tracing_Flag),D);
                    for i=1:N
                        if Tracing_Flag(i)==1               % Less Cats in Tracing Mode, then execute tracing process
                            v(i,:)=v(i,:)+c.*(-1+2.*rand(1,D)).*(Each_Iterate_BestThresholds(Iterates,:)-x(i,:));
                            x_trace(i,:)=x(i,:)+v(i,:);
                            x_trace(i,:)=abs(round(x_trace(i,:)));
                            % boudary detect
                                x_trace(x_trace>nd)=nd;
                                x_trace(x_trace<st)=st;
                            % update
                                UpdatedXtraceFit=fitness(LP,x_trace(i,:));
                                UpdatedXtraceFitness=1/UpdatedXtraceFit;
                                FunCount=FunCount+1;
                                if UpdatedXtraceFitness < Cat_fitness(i)
                                    x(i,:)=x_trace(i,:);
                                    Cat_fitness(i)=UpdatedXtraceFitness;
                                    if fitness(LP,Each_Iterate_BestThresholds(Iterates,:))< UpdatedXtraceFit
                                        Each_Iterate_BestThresholds(Iterates,:)=x(i,:);
                                    end
                                else
                                    x(i,:)=x(i,:);
                                end
                        else                                % Most Cats in Seeking Mode, then execute seeking process
                            % execute Copy process
                                x_i_copy=zeros(SMP,D);
                                if SPC==1
                                    cpp=SMP-1;
                                    x_i_copy(SMP,:)=x(i,:);
                                else
                                    cpp=SMP;
                                end
                                for cp=1:cpp
                                    x_i_copy(cp,:)=x(i,:);
                                end
                            % execute Mutate process
                                xi_copy_fitness=zeros(1,cpp);
                                % Mutate
                                for cp=1:cpp
                                    Modify=randperm(D);
                                    Modify(Modify<=CDC)=1;
                                    Modify(Modify>CDC)=0;
                                    x_i_copy(cp,:)=x(i,:)+(-1+2.*rand(1,D)).*SRD.*Modify.*x(i,:);
                                    x_i_copy(cp,:)=abs(round(x_i_copy(cp,:)));
                                    % boudary detect
                                        x_i_copy(x_i_copy>nd)=nd;
                                        x_i_copy(x_i_copy<st)=st;
                                    xi_copy_fitness(1,cp)=1/fitness(LP,x_i_copy(cp,:));
                                    FunCount=FunCount+1;
                                end
                                % Select(strategy one: 参考书本《群体智能与仿生计算》，selcet the best copy_cat to the next iteration;
                                %        strategy two: while in the original paper Shu-Chuan Chu et.al using the Roulette Wheel Selection)
                                % Strategy One
                                    [xi_best_fitness,xi_best_index]=min(xi_copy_fitness);
                                    if xi_best_fitness < Cat_fitness(i)
                                        x(i,:)=x_i_copy(xi_best_index,:);
                                        Cat_fitness(i)=xi_best_fitness;
                                        if fitness(LP,Each_Iterate_BestThresholds(Iterates,:))< fitness(LP,x(i,:));
                                            Each_Iterate_BestThresholds(Iterates,:)=x(i,:);
                                        end
                                    end
%                                 % Strategy Two
%                                     x_i_copy_cum=zeros(1,cpp);
%                                     for cpc=1:cpp
%                                         x_i_copy_cum(cpc)=sum(xi_copy_fitness(1:cpc));
%                                     end
%                                     Index_select=find((x_i_copy_cum(1)+rand*x_i_copy_cum(end))>=x_i_copy_cum,1,'first');
%                                     if 1/fitness(LP,x_i_copy(Index_select,:)) < Cat_fitness(i)
%                                         x(i,:)=x_i_copy(Index_select,:);
%                                     end                                    
                        end
                    end 
             CSO_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
             EachRunFunCalNum(Iterates)=FunCount;
             Iterates=Iterates+1;
             end % End "while(~Success_FindBest_Index(RepeatTimes) || Iterates<=T)"
%% 记录实验结果
    Each_Iterate_BestFitness(Each_Iterate_BestFitness==0)=10.^9;        %去除零元素
    CSO_result.EachRunEveryIterBestFitness(RepeatTimes,:)=1./Each_Iterate_BestFitness;
    [~,MinFitIndex]=min(Each_Iterate_BestFitness);
    CSO_result.EachRunBestFitness(RepeatTimes)=fitness(LP,Each_Iterate_BestThresholds(MinFitIndex,:));
    CSO_result.EachRunBestThresholds(RepeatTimes,:)=Each_Iterate_BestThresholds(MinFitIndex,:);
    CSO_result.EachRunConvergenceTime(RepeatTimes)=CSO_result.EachRunEveryIterConvergenceTime(RepeatTimes,MinFitIndex);
    CSO_result.EachRunConvergenceFunCalNum(RepeatTimes)=EachRunFunCalNum(MinFitIndex);
end % End "for RepeatTimes=1:RunTimes"
    
disp('Statistic Metrics is Calculating...')
%% 统计实验结果
    % 计算适应度值并统计
    [CSO_result.Fitness.Max,CSOFMax_Index]=max(CSO_result.EachRunBestFitness);
    [CSO_result.Fitness.Min,~]=min(CSO_result.EachRunBestFitness);
    CSO_result.Fitness.Mean=mean(CSO_result.EachRunBestFitness);
    CSO_result.Fitness.Variance=var(CSO_result.EachRunBestFitness);
    CSO_result.BestThresholds=sort(CSO_result.EachRunBestThresholds(CSOFMax_Index,:));
    % 计算“成功查找率”,“平均每次实验收敛时间”并统计
    CSO_result.Success_Rate=sum(Success_FindBest_Index)/RunTimes;
    CSO_result.MeanConvergenceTime=mean(CSO_result.EachRunConvergenceTime);
    
disp('Statistic Metrics is Calculated !')

% 保存CSO结果：ImageNmae_THChar_CSO_result.mat
    FILENAME=strcat(Alg_Name,'_',ImageName,'_',TH_Char,'_CSO_result.mat');
    save(FILENAME,'CSO_result');      
        
disp('the Cat Swarm Optimization is accomplised !!!')
end