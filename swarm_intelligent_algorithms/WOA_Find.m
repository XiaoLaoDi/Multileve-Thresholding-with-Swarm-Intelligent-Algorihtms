function  WOA_result=WOA_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
% function WOA_result=WOA_Find(FunctionName,TH_num,RunTimes,MAX_Iterations)
%    The Whale Optimization Algorithm,proposed by Seyedali Mirjalili and Andrew Lewis in 2015,is try to optimize some minimization problems
%    Calculating the maximum Kapur_Entropy/Between-Variance use the Whale Opimization Algorithm
% Input:
%    FunctionName:the optimized method---Kapur_Entropy/Otsu
%    TH_num:number of thresholds
%    RunTimes:the Repeat Algorithm Running Times
%    MAX_Func_Counts:
% Output:
%    WOA_result: is a 'Struct' containing the following result information
%        WOA_result.Fitness:
%            WOA_result.Fitness.Mean
%            WOA_result.Fitness.Variance
%            WOA_result.Fitness.Max
%            WOA_result.Fitness.Min
%        WOA_result.Success_Rate=sum(Success_FindBest_Num)/RunTimes;
%        WOA_result.BestThresholds: a vector containing 'TH_num' values
%        WOA_result.EachRunConvergenceTime:a vector containing 'RunTimes' values
%        WOA_result.MeanConvergenceTime: the mean convergence time

disp('the Whale Optimization Algorithm is running...')

%% WOA STEPS
% STEP ONE
% Initialize the WOA parameters
global LP nd st BEST_EXHAUSTIVE_FITNESS EPS Gray_image TH_Char ImageName Alg_Name;

N=40;
D=TH_num;
fitness=FunctionName;
LinearConstant=4000;
T=MAX_Iterations;

%% Ԥ�����ڴ�
WOA_result.EachRunBestThresholds=zeros(RunTimes,TH_num);
WOA_result.EachRunBestFitness=zeros(1,RunTimes);
WOA_result.BestThresholds=zeros(1,TH_num);
WOA_result.EachRunConvergenceTime=zeros(1,RunTimes);
WOA_result.EachRunConvergenceFunCalNum=zeros(1,RunTimes);
WOA_result.EachRunEveryIterBestFitness=zeros(RunTimes,T);
WOA_result.EachRunEveryIterConvergenceTime=zeros(RunTimes,T);

%% �����ѭ������WOA��RunTimes��
Success_FindBest_Index=zeros(1,RunTimes);

for RepeatTimes=1:RunTimes
    rng(sum(RepeatTimes*nd*3000), 'twister');
    
	tic
    FunCount=0;
    EachRunFunCalNum=zeros(1,T);
    
    % STEP TWO
    %------Initialize the WOA swarm's position and velocity------------
    %Initialize the positions of search agents
	x=zeros(N,D);
    Fit=zeros(1,N);
	for i=1:N
		for j=1:D
			x(i,j)=st+round((nd-st)*rand);               % randomized position
		end
	end

    % STEP THREE
    %------the WOA main iterations ------------
    Each_Iterate_BestFitness=zeros(1,T);
    Each_Iterate_BestThresholds=zeros(T,D);
    Iterates=1;
    Success_FindBest_Index(RepeatTimes)=0;
    while(~Success_FindBest_Index(RepeatTimes) && Iterates<=T)
        % boundary detect
            x=round(x);
            x(x>nd)=nd;
            x(x<st)=st;
        for i=1:N
            % Calculate objective function for each search agent
            Fit(i)=1/fitness(LP,x(i,:));
            FunCount=FunCount+1;
         end
        
        % iterations
        [Each_Iterate_BestFitness(Iterates),min_index]=min(Fit);
        Each_Iterate_BestThresholds(Iterates,:)=x(min_index,:);
        if (abs(fitness(LP,Each_Iterate_BestThresholds(Iterates,:))-BEST_EXHAUSTIVE_FITNESS)<=EPS)
            WOA_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
            EachRunFunCalNum(Iterates)=FunCount;
            Success_FindBest_Index(RepeatTimes)=1;
            break;
        end   
		
	a=2-Iterates*((2)/LinearConstant); % a decreases linearly fron 2 to 0 in Eq. (2.3)
    
    % a2 linearly dicreases from -1 to -2 to calculate Iterates in Eq. (3.12)
    a2=-1+Iterates*((-1)/LinearConstant);
    
    % Update the Position of search agents 
        for i=1:N
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            A=2*a*r1-a;  % Eq. (2.3) in the paper
            C=r2;      % Eq. (2.4) in the paper
            b=5;               %  parameters in Eq. (2.5)
            l=(a2-1)*rand+1;   %  parameters in Eq. (2.5)
            p = rand();        % p in Eq. (2.6)
            for j=1:D
                if p<0.5   
                    if abs(A)>=1
                        rand_leader_index = floor(N*rand()+1);
                        X_rand = x(rand_leader_index, :);
                        D_X_rand=abs(C*X_rand(j)-x(i,j)); % Eq. (2.7)
                        x(i,j)=X_rand(j)-A*D_X_rand;      % Eq. (2.8)
                    elseif abs(A)<1
                        D_Leader=abs(C*Each_Iterate_BestThresholds(Iterates,j)-x(i,j)); % Eq. (2.1)
                        x(i,j)=Each_Iterate_BestThresholds(Iterates,j)-A*D_Leader;      % Eq. (2.2)
                    end
                elseif p>=0.5
                    distance2Leader=abs(Each_Iterate_BestThresholds(Iterates,j)-x(i,j));
                    % Eq. (2.5)
                    x(i,j)=distance2Leader*exp(b.*l).*cos(l.*2*pi)+Each_Iterate_BestThresholds(Iterates,j); 
                end    
            end
        end
        WOA_result.EachRunEveryIterConvergenceTime(RepeatTimes,Iterates)=toc;
        EachRunFunCalNum(Iterates)=FunCount;
        Iterates=Iterates+1;
    end % End "while(~Success_FindBest_Index(RepeatTimes) || Iterates<=T)"

%% ��¼ʵ����
    Each_Iterate_BestFitness(Each_Iterate_BestFitness==0)=10.^9;        %ȥ����Ԫ��
    WOA_result.EachRunEveryIterBestFitness(RepeatTimes,:)=1./Each_Iterate_BestFitness;
    [~,MinFitIndex]=min(Each_Iterate_BestFitness);
    WOA_result.EachRunBestFitness(RepeatTimes)=fitness(LP,Each_Iterate_BestThresholds(MinFitIndex,:));
    WOA_result.EachRunBestThresholds(RepeatTimes,:)=Each_Iterate_BestThresholds(MinFitIndex,:);
    WOA_result.EachRunConvergenceTime(RepeatTimes)=WOA_result.EachRunEveryIterConvergenceTime(RepeatTimes,MinFitIndex);
    WOA_result.EachRunConvergenceFunCalNum(RepeatTimes)=EachRunFunCalNum(MinFitIndex);
end % End "for RepeatTimes=1:RunTimes"
    
disp('Statistic Metrics is Calculating...')
%% ͳ��ʵ����
    % ������Ӧ��ֵ��ͳ��
    [WOA_result.Fitness.Max,WOAFMax_Index]=max(WOA_result.EachRunBestFitness);
    [WOA_result.Fitness.Min,~]=min(WOA_result.EachRunBestFitness);
    WOA_result.Fitness.Mean=mean(WOA_result.EachRunBestFitness);
    WOA_result.Fitness.Variance=var(WOA_result.EachRunBestFitness);
    WOA_result.BestThresholds=sort(WOA_result.EachRunBestThresholds(WOAFMax_Index,:));
    % ���㡰�ɹ������ʡ�,��ƽ��ÿ��ʵ������ʱ�䡱��ͳ��
    WOA_result.Success_Rate=sum(Success_FindBest_Index)/RunTimes;
    WOA_result.MeanConvergenceTime=mean(WOA_result.EachRunConvergenceTime);
    
disp('Statistic Metrics is Calculated !')

% ����WOA�����ImageNmae_THChar_WOA_result.mat
    FILENAME=strcat(Alg_Name,'_',ImageName,'_',TH_Char,'_WOA_result.mat');
    save(FILENAME,'WOA_result');     

disp('the Whale Optimization Algorithm is accomplised !!!')



end