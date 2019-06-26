library(tidyverse)
library(forcats)
library(magrittr)

data = read_csv('output/from_example_data/output_psII_level0.csv',
                na = 'nan')

data %<>% mutate(exp = as_factor(exp),
                 exp = fct_rev(exp),
                 gtype = as_factor(gtype),
                 gtype = fct_relevel(gtype,sort),
                 gtype = fct_relevel(gtype,'WT',after=0),
                 measurement = case_when(parameter == 'FvFm' ~ 'FvFm',
                                         TRUE ~ 'IndC'),
                 parameter = as_factor(parameter))


summ = data %>% 
  filter(frame %in% c('Fm','Fmp')) %>% 
  group_by(gtype, exp, date, parameter, measurement) %>% 
  summarise(avg = mean(yii_avg),
            std = mean(yii_avg))

ggplot(summ %>% filter(parameter == 'FvFm')) +
  geom_col(aes(y = avg, x=gtype))+
  facet_wrap(~parameter)

ggplot() + 
  geom_line(data = summ %>% filter(parameter != 'FvFm'), aes(x = parameter, y = avg, group = interaction(exp,gtype), linetype = interaction(exp), color = gtype), size=1)+
  geom_point(data = summ %>% filter(parameter == 'FvFm'), aes(x = parameter, y = avg, color = gtype, shape = interaction(exp)))+
  scale_x_discrete(drop = FALSE)+
  labs(title = 'yii')
