fig6 = figure()
subplot(1,2,1),plot(aq');
title('$$aq_{in}$$ (before inverse dynamics)','interpreter', 'latex');
grid on;
subplot(1,2,2),plot(qddr);
grid on;
title('$$aq_{out}$$ (after froward dynamics)','interpreter', 'latex');