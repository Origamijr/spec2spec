function [y, yworad] = synthesize_voice_2portscatter(glottal_flow, S)

x = glottal_flow;
Ns = length(x);
NSECT = length(S);
Nj = NSECT - 1;      % number of junctions
k = (S(1:end-1) - S(2:end)) ./ (S(1:end-1) + S(2:end));

% left boundary
R0 = .9;       

% right boundary
BRL = -[0.2162 0.2171 0.0545];
ARL = [1 -0.6032 0.0910];
state = [0 0];

% amplitude-complementary transmission filter
BTL = ARL + BRL;
ATL = ARL;
stateTL = [0 0];

% initialization
ins = zeros(Nj, 2);  % upper and lower inputs
outs = zeros(Nj, 2); % upper and lower outputs
lend = 0; % lower end
uend = 0; % upper end

for n = 3:Ns
    for m = 1:Nj
        if m == 1
            ins(1, 1) = lend*R0 + x(n-1); %lend (previous time sample)
        else
            ins(m, 1) = outs(m-1, 1);
        end
        if m == Nj
            [ins(m, 2), state] = filter(BRL, ARL, uend, state);
        else
            ins(m, 2) = outs(m+1, 2);
        end
    end
    
    % calculate port outputs and end points
    [y(n), stateTL] = filter(BTL, ATL, outs(Nj, 1), stateTL); % output
    yworad(n) = outs(Nj, 1);
    uend = outs(Nj, 1); % upper end
    lend = outs(1, 2);  % lower end
    
    for m = 1:Nj
        outs(m, 1) = ins(m, 1)*(1+k(m)) + ins(m, 2)*(-k(m)); % upper
        outs(m, 2) = ins(m, 2)*(1-k(m)) + ins(m, 1)*k(m);    % lower
    end
end
y = y';
yworad = yworad';