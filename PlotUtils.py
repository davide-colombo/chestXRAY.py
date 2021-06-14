
# dict(alpha = 0.5, bins = 100, density = True, stacked = True)     # kwargs for matplotlib.pyplot
# plt.hist(class1, **kwargs, color='g', label='bacteria')
# plt.hist(class2, **kwargs, color='y', label='normal')
# plt.hist(class3, **kwargs, color='r', label='virus')


from typing import Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns

class PlotUtils:

    CLASS_COLOR = {'bacteria': 'dodgerblue', 'normal': 'orange', 'virus': 'deeppink'}

    def plot_class_histogram(self, class_stats: List[List[float]],
                             class_names: List[str] = ('bacteria', 'normal', 'virus'),
                             set_style   = True, sns_style: str = 'white',
                             set_figsize = True, figsize: Tuple = (10, 10),
                             limit_xaxis = True, xlim: Tuple[float, float] = (0, 250),
                             set_title   = True, title: str  = ' ',
                             set_ylabel  = True, ylabel: str = ' ',
                             set_xlabel  = True, xlabel: str = ' ',
                             show_legend = True,
                             **kwargs):

        if set_style:
            sns.set_style(sns_style)
        if set_figsize:
            plt.figure(figsize = figsize, dpi = 80)

        for i in range(0, len(class_stats)):
            sns.distplot(class_stats[i], color = self.CLASS_COLOR.get(class_names[i]), label = class_names[i],  **kwargs)

        if set_title:
            plt.title(title)
        if set_ylabel:
            plt.ylabel(ylabel)
        if set_xlabel:
            plt.xlabel(xlabel)
        if limit_xaxis:
            plt.xlim(xlim)
        if show_legend:
            plt.legend()
