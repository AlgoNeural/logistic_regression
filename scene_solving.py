# This software uses the community version of the manim animation library for python
# https://github.com/ManimCommunity/manim
# The manim license can be found in manim-license.txt.
import math
from sympy import sin, cos, exp, Symbol, lambdify, latex, print_latex
import scipy

import numpy as np
from manim import *
from scipy import stats
from sklearn.linear_model import LogisticRegression

from manim.mobject.geometry.tips import ArrowTriangleFilledTip

COLOR_SUCCESS = GREEN
COLOR_FAIL = RED
COLOR_UNKNOWN = YELLOW
COLOR_WRONG_LABEL = PURPLE


stroke_width_line05 = 0.9


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_sigmoid_curve(ax, x_range, horizontal_squish=1):
    return ParametricFunction(lambda t: ax.c2p(t, sigmoid(t * horizontal_squish)), t_range=x_range)

def get_curve(fcn, ax, x_range):
    return ParametricFunction(lambda t: ax.c2p(t, fcn(t)), t_range=x_range)

def get_func_graph(fcn, x_range, y_range, x_length, y_length, scale, font_size, formula_shift_up=0,
                   formula_shift_left=0, include_equation=True, curve_length_x_axis_ratio=1):
    arrow_tip_scale = 0.15
    ax = get_axes(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length, scale=scale,
                  arrow_tip_scale=arrow_tip_scale)
    x_center = (x_range[1] + x_range[0]) / 2
    x_diff = (x_range[1] - x_range[0]) / 2
    logistic_fcn = get_curve(fcn, ax=ax, x_range=[x_center - x_diff * curve_length_x_axis_ratio,
                                                  x_center + x_diff * curve_length_x_axis_ratio])
    graph_dict = {"ax": ax, "fcn": logistic_fcn}
    graph_vgroup = VGroup(ax, logistic_fcn)
    if include_equation:
        equation = MathTex(r'f(x) = \frac{1}{1 + e^{-x}}', font_size=font_size). \
            move_to(ax.get_center() + LEFT * formula_shift_left + UP * formula_shift_up).set_color(YELLOW)
        graph_dict["equation"] = equation
        graph_vgroup.add(equation)

    return graph_vgroup, graph_dict

class FilledArrowAnimated:
    def __init__(self, arrow_tracker, width=2, tip_height=0.8, shaft_length=5, tip_length=2.5, scale=1,
                 move_to=np.array([0, 0, 0]), color=WHITE):
        def get_arrow(fill_percent):
            fill_percent = fill_percent % 1

            def get_poly_points(pos):
                position_list = []
                for key, val in pos.items():
                    position_list.append(val + [0])
                return position_list

            def get_filled_shaft(fraction, pos):
                right_x = pos["A"][0] + shaft_length * fraction
                new_pos = {"A": pos["A"],
                           "B": [right_x, width/2],
                           "F": [right_x, -width/2],
                           "G": pos["G"]}
                new_position_list = get_poly_points(new_pos)
                return Polygon(*new_position_list, color=color, fill_color=color, fill_opacity=1)

            def get_filled_tip(fraction, pos):
                print("tip fraction: " + str(fraction))
                new_pos = {"C": pos["C"],
                           "G": list(fraction * np.array(pos["D"]) + (1 - fraction) * np.array(pos["C"])),
                           "H": list(fraction * np.array(pos["D"]) + (1 - fraction) * np.array(pos["E"])),
                           "E": pos["E"]}
                new_position_list = get_poly_points(new_pos)
                return Polygon(*new_position_list, color=color, fill_color=color, fill_opacity=1)

            pos = {"A": [-shaft_length, width / 2],
                   "B": [0, width / 2],
                   "C": [0, width / 2 + tip_height],
                   "D": [tip_length, 0],
                   "E": [0, -width / 2 - tip_height],
                   "F": [0, -width / 2],
                   "G": [-shaft_length, -width / 2]}

            position_list = get_poly_points(pos)
            arrow = Polygon(*position_list, color=color)
            total_length = shaft_length + tip_length
            filled_shaft_fraction = min(1, fill_percent * total_length / shaft_length)
            arrow_return = VGroup(arrow)
            filled_shaft = get_filled_shaft(filled_shaft_fraction, pos)
            arrow_return.add(filled_shaft)
            tip_fraction = (total_length * fill_percent - shaft_length) / tip_length
            if tip_fraction > 0:
                filled_tip = get_filled_tip(tip_fraction, pos)
                arrow_return.add(filled_tip)
            return arrow_return.scale(scale).move_to(move_to)
        arrow = always_redraw(lambda: get_arrow(arrow_tracker.get_value()))
        self.arrow = arrow

    def get(self):
        return self.arrow


def align_top(obj, template):
    current_coords = obj.get_center()
    x = current_coords[0]
    y = current_coords[1]
    z = current_coords[2]
    shift = template.get_top()[1] - obj.get_top()[1]
    obj.move_to(np.array([x, y + shift, z]))
    return obj


def align_bottom(obj, template):
    current_coords = obj.get_center()
    x = current_coords[0]
    y = current_coords[1]
    z = current_coords[2]
    shift = template.get_bottom()[1] - obj.get_bottom()[1]
    obj.move_to(np.array([x, y + shift, z]))
    return obj


def align_left(obj, template):
    current_coords = obj.get_center()
    x = current_coords[0]
    y = current_coords[1]
    z = current_coords[2]
    shift = template.get_left()[0] - obj.get_left()[0]
    obj.move_to(np.array([x + shift, y, z]))
    return obj


def align_vertically(mobj, template):
    diff = template.get_center()[1] - mobj.get_center()[1]
    mobj.shift(UP * diff)


def get_ghost(mobj, opacity=0.8):
    ghost = mobj.copy()
    ghost.set_opacity(opacity)
    return ghost


def get_axes_config(arrow_tip_scale):
    axis_config = {
        "include_ticks": False,
        "color": BLUE
    }
    if arrow_tip_scale is not None:
        axis_config["tip_length"] = arrow_tip_scale
        axis_config["tip_width"] = arrow_tip_scale
    return axis_config


def get_axes(x_range, y_range, x_length, y_length, scale, arrow_tip_scale=None):
    axis_config = get_axes_config(arrow_tip_scale)

    return Axes(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length,
                axis_config=axis_config,
                ).scale(scale)


def get_3d_axes(x_range, y_range, z_range, x_length, y_length, z_length, scale, arrow_tip_scale=None):
    axis_config = get_axes_config(arrow_tip_scale)

    return ThreeDAxes(x_range=x_range, y_range=y_range, z_range=z_range, x_length=x_length, y_length=y_length,
                z_length=z_length, axis_config=axis_config,
                ).scale(scale)





def give_rainbow_colors(curve, shift=0):
    raw_color_array = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]

    def shift_right(ar):
        temp = ar[-1]
        for i in range(len(ar) - 1, 0, -1):
            ar[i] = ar[i - 1]
        ar[0] = temp
        return ar

    new_array = raw_color_array
    for _ in range(shift):
        new_array = shift_right(new_array)

    return curve.set_stroke_color(new_array)


def get_student_label(stud, result, numeric=False, font_size=40):
    stroke_width = 3
    if result == 0:
        if numeric:
            label = MathTex(r"0", color=COLOR_FAIL, font_size=font_size, stroke_width=stroke_width)
        else:
            label = Cross(color=COLOR_FAIL, stroke_width=6).scale(0.13).set_color(COLOR_FAIL)
    elif result == 1:
        if numeric:
            label = MathTex(r"1", color=COLOR_SUCCESS, font_size=font_size, stroke_width=stroke_width)
        else:
            label = MathTex(r"\checkmark", color=COLOR_SUCCESS, font_size=font_size, stroke_width=stroke_width)
    elif result is None:
        label = MathTex(r"?", color=COLOR_UNKNOWN, font_size=font_size, stroke_width=stroke_width)
    else:
        raise ValueError("Wrong result type!")
    return label.move_to(stud.get_center()).set_z_index(1)


# def get_func_graph(a, b, x_range, y_range, x_length, y_length, scale, type='linear'):
#     arrow_tip_scale = 0.12
#     ax = get_axes(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length, scale=scale,
#                   arrow_tip_scale=arrow_tip_scale)
#     if type == 'linear':
#         my_func = lambda t: a * t + b
#     elif type == 'sigmoid':
#         my_func = lambda t: 1 / (1 + np.exp(-(a * t + b)))
#     else:
#         raise ValueError('Wrong type!')
#     linear_fcn = ParametricFunction(lambda t: ax.c2p(t, my_func(t)), t_range=[x_range[0] * 0.8, x_range[1] * 0.8],
#                                     stroke_width=2)
#     return VGroup(ax, linear_fcn), {"ax": ax, "fcn": linear_fcn}


def get_coordinate_line(scene_obj, x_range, y_range, x_length, y_length, scale, arrow_tip_scale=None):
    axes = get_axes(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length, scale=scale,
                    arrow_tip_scale=arrow_tip_scale)
    axes_temp = axes.copy().set_opacity(0)
    unit_vec_y = axes.c2p(0, 1, 0) - axes.c2p(0, 0, 0)
    y_coord_line = axes.submobjects[1]
    scene_obj.remove(axes.submobjects[1])
    del axes.submobjects[1]
    coord_line_dict = {"y_coord_line": y_coord_line, "unit_vec_y": unit_vec_y, "original_coords": axes_temp}
    all_obj_vgoup = VGroup(axes, axes_temp)
    return axes, all_obj_vgoup, coord_line_dict


class Intro(ThreeDScene):

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_sigmoid_curve(self, ax, x_range, horizontal_squish=1):
        return ParametricFunction(lambda t: ax.c2p(t, self.sigmoid(t * horizontal_squish)), t_range=x_range)

    def get_logistic_func(self, x_range, y_range, x_length, y_length, scale, font_size, formula_shift_up,
                          formula_shift_left):
        arrow_tip_scale = 0.15
        ax = get_axes(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length, scale=scale,
                      arrow_tip_scale=arrow_tip_scale)
        logistic_fcn = self.get_sigmoid_curve(ax=ax, x_range=x_range)
        equation = MathTex(r'f(x) = \frac{1}{1 + e^{-x}}', font_size=font_size). \
            move_to(ax.get_center() + LEFT * formula_shift_left + UP * formula_shift_up).set_color(YELLOW)

        return VGroup(ax, logistic_fcn, equation), {"ax": ax, "logistic_fcn": logistic_fcn, "equation": equation}

    def get_linear_func(self, a, b, x_range, y_range, x_length, y_length, scale):
        arrow_tip_scale = 0.26
        ax = get_axes(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length, scale=scale,
                      arrow_tip_scale=arrow_tip_scale)
        linear_fcn = ParametricFunction(lambda t: ax.c2p(t, a * t + b), t_range=x_range)
        return VGroup(ax, linear_fcn), {"ax": ax, "fcn": linear_fcn}

    def get_human(self):
        return ImageMobject("data/penguin.png").scale(0.15)

    def get_people(self):
        n_rows = 4
        people = Group()
        people_list = []
        last_position = None
        first_position_in_prev_row = None
        last_z_index = 0
        n_in_row = 2
        horizontal_shift = 0.8
        vertical_shift = 0.3
        for i in range(n_rows):
            for j in range(n_in_row):
                last_z_index = last_z_index - 1
                human = self.get_human().set_z_index(last_z_index)
                if last_position is not None:
                    if j != 0:
                        human.move_to(last_position + RIGHT * horizontal_shift)
                    elif first_position_in_prev_row is not None:
                        human.move_to(first_position_in_prev_row + LEFT * horizontal_shift / 2 + UP * vertical_shift)
                last_position = human.get_center()
                people.add(human)
                people_list.append(human)
                if j == 0:
                    first_position_in_prev_row = last_position
            n_in_row += 1
        return people, people_list

    def construct(self):
        top_elements_height = 2

        # Sigmoid
        x_range = [-4.5, 4.5]
        y_range = [-0.5, 1.5]
        x_length = 5
        y_length = 3
        logistic_graph, logistic_graph_dict = self.get_logistic_func(x_range=x_range, y_range=y_range,
                                                                     x_length=x_length, y_length=y_length,
                                                                     scale=2, font_size=55, formula_shift_up=1,
                                                                     formula_shift_left=2.5)
        logistic_graph_colorful, logistic_graph_colorful_dict = self.get_logistic_func(x_range=x_range, y_range=y_range,
                                                                                       x_length=x_length,
                                                                                       y_length=y_length,
                                                                                       scale=2, font_size=55,
                                                                                       formula_shift_up=1,
                                                                                       formula_shift_left=2.5)
        give_rainbow_colors(logistic_graph_colorful_dict["logistic_fcn"])
        logistic_graph_small, logistic_graph_small_dict = self.get_logistic_func(x_range=x_range, y_range=y_range,
                                                                                 x_length=x_length, y_length=y_length,
                                                                                 scale=0.4, font_size=32,
                                                                                 formula_shift_up=0.5,
                                                                                 formula_shift_left=1.5)
        logistic_graph_small.shift(LEFT * 4 + UP * (top_elements_height + 0.35))
        give_rainbow_colors(logistic_graph_small_dict["logistic_fcn"])

        # Pierre
        pierre_img = ImageMobject("data/Pierre_Francois_Verhulst.jpeg").scale(0.8).shift(RIGHT * 4.5 +
                                                                                         UP * top_elements_height)

        # Population growth
        people, people_list = self.get_people()
        people.move_to(pierre_img.get_center() + DOWN * 4).scale(0.7)
        humans_to_reveal_at_once = 2
        n_generations = 4
        next_person_to_appear = 0
        generation_anims = []
        for _ in range(n_generations):
            next_gen = [people[i] for i in range(next_person_to_appear,
                                                 next_person_to_appear + humans_to_reveal_at_once)]
            anims = [FadeIn(person) for person in next_gen]
            generation_anims.append(anims)
            humans_to_reveal_at_once += 1
            next_person_to_appear = next_person_to_appear + len(next_gen)

        # Logistic regression
        logistic_regression_text = Tex(r'\begin{minipage}{15cm}Logistic regression\end{minipage}', font_size=45). \
            shift(DOWN * 3.7 + LEFT * 2.5)

        data_x = np.array([-4, -2.3, -1.5, 1.1, 2, 3, 3.7])
        data_y = np.array([0, 0, 0, 1, 1, 1, 1])
        res = stats.linregress(data_x, data_y)
        linear_fcn, linear_fcn_dict = self.get_linear_func(res.slope, res.intercept, x_range, y_range=[-1.5, 1.8],
                                                           x_length=x_length,
                                                           y_length=2.5, scale=1.4)
        linear_fcn.shift(LEFT + DOWN * 0.75)
        logistic_regression_text.next_to(linear_fcn, DOWN * 0.2)
        give_rainbow_colors(linear_fcn_dict["fcn"])
        points = [Circle(radius=0.05, fill_color=RED, color=RED, fill_opacity=1, stroke_width=2)
                  .move_to(linear_fcn_dict["ax"].c2p(x_, y_)) for x_, y_ in zip(data_x, data_y)]
        point_creation_time = 0.7
        time_to_create_all_points = 2.5
        start_times = np.random.random(len(points)) * (time_to_create_all_points - point_creation_time)
        point_anims = [Succession(Wait(run_time=st), FadeIn(p, run_time=point_creation_time),
                                  Wait(run_time=time_to_create_all_points - point_creation_time - st))
                       for p, st in zip(points, start_times)]
        logistic_curve_to_transform_into = self.get_sigmoid_curve(linear_fcn_dict["ax"], x_range, horizontal_squish=2)
        give_rainbow_colors(logistic_curve_to_transform_into)

        ################################################################################################

        # self.play(Create(logistic_graph), run_time=3.5)
        self.play(Create(logistic_graph_dict["ax"]), run_time=1)
        self.play(Write(logistic_graph_dict["logistic_fcn"]), Write(logistic_graph_dict["equation"]), run_time=2.5)

        self.play(Transform(logistic_graph, logistic_graph_colorful), run_time=2.5)
        self.wait(1)
        # 7s

        self.play(Transform(logistic_graph, logistic_graph_small), FadeIn(pierre_img), run_time=4)
        self.wait(5)

        for anims in generation_anims:
            self.play(*anims, run_time=3/4)
        # 20s

        self.wait(7)
        self.play(Write(logistic_regression_text), run_time=3)
        # 30s

        self.wait(1.5)
        self.play(*point_anims) # 2.5s
        self.play(Create(linear_fcn), run_time=2)
        logistic_ghost = get_ghost(logistic_graph_dict["logistic_fcn"])
        self.play(logistic_ghost.animate.move_to(logistic_curve_to_transform_into.get_center()), run_time=2)
        self.remove(logistic_ghost)
        self.play(Transform(linear_fcn_dict["fcn"], logistic_curve_to_transform_into), run_time=2)
        # 40s

        logistic_function_with_title = VGroup(*([linear_fcn, logistic_regression_text] + points))
        logistic_function_with_title_small = logistic_function_with_title.copy().scale(0.6).shift(DOWN * 1.5)
        question_mark = Tex(r'\begin{minipage}{15cm}?\end{minipage}', font_size=140). \
            next_to(logistic_function_with_title_small, UP * 1.7)
        give_rainbow_colors(question_mark)
        question_mark2 = question_mark.copy().set_color(YELLOW)
        question_mark3 = question_mark.copy().set_color(ORANGE)
        question_mark4 = question_mark.copy().set_color(RED)
        question_mark5 = question_mark.copy().set_color(GREEN)
        question_mark6 = question_mark.copy().set_color(PURPLE)
        self.wait(2)
        self.play(Transform(logistic_function_with_title, logistic_function_with_title_small))
        self.play(Write(question_mark), run_time=1.5)
        self.wait(1)
        color_change_time = 4/5
        self.play(Transform(question_mark, question_mark2), rate_func=linear, run_time=color_change_time)
        self.play(Transform(question_mark, question_mark3), rate_func=linear, run_time=color_change_time)
        self.play(Transform(question_mark, question_mark4), rate_func=linear, run_time=color_change_time)
        self.play(Transform(question_mark, question_mark5), rate_func=linear, run_time=color_change_time)
        self.play(Transform(question_mark, question_mark6), rate_func=linear, run_time=color_change_time)

        self.play(*[FadeOut(mobj) for mobj in self.mobjects], run_time=1.5)

        # 50s


class Exam(ThreeDScene):

    def __init__(self):
        super().__init__()
        self.alpha1 = None
        self.alpha2 = None
        self.alpha3 = None
        self.scene_time = 0

    def get_student(self):
        student_array = ImageMobject("data/student.png").get_pixel_array()
        student_array = np.flip(student_array, axis=1)
        return ImageMobject(student_array).scale(0.15)

    def get_book(self):
        return ImageMobject("data/book.png").scale(0.3)

    def get_all_students(self):
        students = Group()
        shift = 0
        shift_inc = 1
        for i in range(5):
            student = self.get_student().shift(RIGHT * shift)
            students.add(student)
            shift += shift_inc
        return students

    def get_student_circle(self, radius=0.2):
        return Circle(radius=radius, color=WHITE, fill_opacity=0, stroke_opacity=1, stroke_width=2.5)

    def get_student_name(self, stud, name):
        return Tex(r'\begin{minipage}{15cm}' + name + r'\end{minipage}', font_size=22).next_to(stud, UP * 0.5). \
            set_z_index(1)

    def construct(self):
        v_shift = 0.4
        vertical_shift = 2
        font_size = 25
        eq1 = MathTex(r'\sum_{i=1}^\infty x_i^2 \sin(y_i^3)', font_size=font_size, color=YELLOW). \
            shift(UP * (vertical_shift + 0.7))
        eq2 = MathTex(r'\int_\Omega w^\gamma \log(\omega) d\omega', font_size=font_size, color=BLUE). \
            next_to(eq1, RIGHT)
        eq3 = MathTex(r'\nabla_x \langle \psi(x), \phi(x) \rangle', font_size=font_size, color=PURPLE). \
            move_to((eq1.get_center() + eq2.get_center()) / 2 + DOWN * 1.1)
        equation_center = (eq1.get_center() + eq2.get_center() + eq3.get_center()) / 3
        diff1 = eq1.get_center() - equation_center
        diff2 = eq2.get_center() - equation_center
        diff3 = eq3.get_center() - equation_center
        r1 = np.linalg.norm(diff1)
        r2 = np.linalg.norm(diff2)
        r3 = np.linalg.norm(diff3)
        self.alpha1 = np.arctan(diff1[1] / diff1[0])
        self.alpha2 = np.arctan(diff2[1] / diff2[0])
        self.alpha3 = np.arctan(diff3[1] / diff3[0])
        eq_vertical_offset = 0.9
        eq1_final_location = equation_center + UP * eq_vertical_offset
        eq2_final_location = equation_center
        eq3_final_location = equation_center + DOWN * eq_vertical_offset

        def normalize_vec(v):
            v_length = np.linalg.norm(v)
            if v_length > 0.03:
                return v / v_length
            else:
                return np.array([0, 0, 0])

        def get_equation_updater(idx):
            def rotate_eq(eq, dt):
                alpha = None
                r = None
                self.scene_time += dt / 3  # Because there are 3 updater functions.
                max_time = 4
                if self.scene_time < max_time:
                    if idx == 1:
                        self.alpha1 += dt
                        alpha = self.alpha1
                        r = r1
                    elif idx == 2:
                        self.alpha2 += dt
                        alpha = self.alpha2
                        r = r2
                    elif idx == 3:
                        self.alpha3 += dt
                        alpha = self.alpha3
                        r = r3
                    else:
                        raise ValueError("Wrong equation idx!!!!")
                    x_unit = np.array([1, 0, 0])
                    y_unit = np.array([0, 1, 0])
                    eq.move_to(equation_center + r * np.cos(alpha) * x_unit + r * np.sin(alpha) * y_unit)
                else:
                    dir = None
                    if idx == 1:
                        dir = normalize_vec(eq1_final_location - eq1.get_center())
                        eq1.shift(dir * dt)
                    if idx == 2:
                        dir = normalize_vec(eq2_final_location - eq2.get_center())
                        eq2.shift(dir * dt)
                    if idx == 3:
                        dir = normalize_vec(eq3_final_location - eq3.get_center())
                        eq3.shift(dir * dt)

            return rotate_eq

        rotate_eq1 = get_equation_updater(1)
        rotate_eq2 = get_equation_updater(2)
        rotate_eq3 = get_equation_updater(3)

        eq1.add_updater(rotate_eq1)
        eq2.add_updater(rotate_eq2)
        eq3.add_updater(rotate_eq3)
        students = self.get_all_students().shift(LEFT * 6 + UP * vertical_shift)
        stud_anims = [FadeIn(stud) for stud in students]
        book = self.get_book().shift(RIGHT * 4.5 + UP * vertical_shift)

        x_range = [0, 9.8]
        y_range = [0, 1.75]
        x_length = 5.5
        y_length = 2
        hours_studied_graph, hours_studied_vgroup, hours_studied_graph_dict = get_coordinate_line(scene_obj=self,
                                                                                                  x_range=x_range,
                                                                                                  y_range=y_range,
                                                                                                  x_length=x_length,
                                                                                                  y_length=y_length,
                                                                                                  scale=1.2,
                                                                                                  arrow_tip_scale=0.2)
        x_label = Tex(r'\begin{minipage}{15cm}Hours of study\end{minipage}', font_size=35). \
            next_to(hours_studied_graph, DOWN).shift(RIGHT * 2)
        y_axis = hours_studied_graph_dict["y_coord_line"]
        y_label = Tex(r'y', font_size=35).next_to(y_axis, LEFT * 0.3).shift(UP * 0.2)
        all_parts_of_graph = VGroup(hours_studied_vgroup, y_axis, x_label)
        study_hours = [0.3, 1.3, 2.5, 3.7, 5, 6, 7, 8.5]
        names = ["Bob", "Alice", "Nick", "Marta", "Rob", "Carol", "Max", "Lea"]
        student_circles = [self.get_student_circle().move_to(hours_studied_graph_dict["original_coords"].
                                                             c2p(hour, 0, 0)).shift(DOWN) for hour in study_hours]
        all_parts_of_graph.shift(DOWN)
        line_start = hours_studied_graph.get_left() + hours_studied_graph_dict["unit_vec_y"]
        line_end = hours_studied_graph.get_right() + hours_studied_graph_dict["unit_vec_y"]
        line_at_level_1 = DashedLine(start=line_start, end=line_end, color=YELLOW, stroke_width=stroke_width_line05)
        y_equals_1_label = Tex(r'y = 1', font_size=35, color=YELLOW).next_to(line_at_level_1, RIGHT)
        results = [0, None, 0, 0, 1, 1, None, 1]
        student_names = [self.get_student_name(stud_circle, name) for stud_circle, name in zip(student_circles, names)]
        student_circles = VGroup(*student_circles)
        student_names = VGroup(*student_names)
        all_parts_of_graph.add(student_circles)
        all_parts_of_graph.add(student_names)
        study_hours_anims = [Create(stud) for stud in student_circles] + [Write(name) for name in student_names]
        result_labels = [get_student_label(stud, result) for stud, result in zip(student_circles, results)]
        result_labels_numeric = [get_student_label(stud, result, numeric=True)
                                 for stud, result in zip(student_circles, results)]
        result_labels_anims = [Write(lab) for lab in result_labels]
        results_known = [result_labels_anims[i] for i in range(len(results)) if results[i] is not None]
        results_unknown = [result_labels_anims[i] for i in range(len(results)) if results[i] is None]
        stud_circle_removals = [Uncreate(stud) for stud in student_circles]
        stud_circle_removals_known = [stud_circle_removals[i] for i in range(len(results)) if results[i] is not None]
        stud_circle_removals_unknown = [stud_circle_removals[i] for i in range(len(results)) if results[i] is None]
        vert_unit_vec = hours_studied_graph_dict["original_coords"].c2p(0, 1, 0) - \
                        hours_studied_graph_dict["original_coords"].c2p(0, 0, 0)
        ones_anims = [label.animate.shift(vert_unit_vec) for label, result in zip(result_labels, results)
                      if result == 1]
        ones_anims_names = [name.animate.shift(vert_unit_vec) for name, result in zip(student_names, results)
                            if result == 1]
        for i in range(len(results)):
            if results[i] == 1:
                result_labels_numeric[i].shift(vert_unit_vec)
        turn_to_ones_anims = [Transform(label, label_num) for label, label_num, result in zip(result_labels,
                                                                                              result_labels_numeric,
                                                                                              results) if result == 1]
        turn_to_zeros_anims = [Transform(label, label_num) for label, label_num, result in zip(result_labels,
                                                                                               result_labels_numeric,
                                                                                               results) if result == 0]
        destroy_anims = [Unwrite(eq1), Unwrite(eq2), Unwrite(eq3)] + [FadeOut(stud) for stud in students] + \
                        [FadeOut(book)]

        graph_x_range = [-4, 4]
        graph_y_range = [-3, 3]
        graph_x_length = 2.8
        graph_y_length = 1.9
        graph_scale = 1
        linear_fcn, linear_fcn_dict = get_func_graph(a=0.6, b=0.5, x_range=graph_x_range, y_range=graph_y_range,
                                                     x_length=graph_x_length,
                                                     y_length=graph_y_length, scale=graph_scale)
        linear_fcn.move_to(np.array([-4.5, 2.5, 0]))
        equation = MathTex(r'y = ax + b', font_size=24, color=YELLOW).next_to(linear_fcn_dict["ax"], RIGHT * 0.1). \
            shift(UP * 1 + LEFT * 0.5)
        axis_labels = linear_fcn_dict["ax"].get_axis_labels(x_label='x', y_label='y')
        label_scale = 0.75
        axis_labels[0].scale(label_scale)
        axis_labels[1].scale(label_scale).shift(LEFT * 0.5)
        linear_fcn.add(equation)
        linear_fcn.add(axis_labels)
        linear_fcn.shift(DOWN * v_shift)
        x_data = []
        y_data = []
        for result, hour in zip(results, study_hours):
            if result is not None:
                x_data.append(hour)
                y_data.append(result)

        res = stats.linregress(np.array(x_data), np.array(y_data))
        fitted_straight_line = ParametricFunction(lambda t: hours_studied_graph_dict["original_coords"].
                                                  c2p(t, res.slope * t + res.intercept), t_range=x_range)
        fitted_straight_line = give_rainbow_colors(fitted_straight_line)
        linear_fcn_dict["fcn"] = give_rainbow_colors(linear_fcn_dict["fcn"])
        straight_line_to_transform = linear_fcn_dict["fcn"].copy()
        straight_line_to_transform.set_opacity(0.5)
        straight_line_to_transform = give_rainbow_colors(straight_line_to_transform)
        results_on_line_y = [res.slope * x + res.intercept for x in study_hours]
        stud_positions_on_line = [hours_studied_graph_dict["original_coords"].
                                  c2p(x, y) for x, y in zip(study_hours, results_on_line_y)]

        ################################################################################################
        equation_write_time = 0.5
        self.play(Write(eq1), run_time=equation_write_time)
        self.play(Write(eq2), run_time=equation_write_time)
        self.play(Write(eq3), run_time=equation_write_time)

        for anim in stud_anims: # 5 students
            self.play(anim, run_time=0.3)
        self.play(FadeIn(book), run_time=1)
        # 4s

        self.play(Create(hours_studied_graph), Write(x_label), run_time=1)
        self.play(*study_hours_anims, run_time=1.5)
        self.wait(1)
        self.play(*(results_known + stud_circle_removals_known), run_time=2)
        self.wait(2)
        self.play(*(results_unknown + stud_circle_removals_unknown), run_time=2) # 13.5s
        self.wait(0.5)
        self.play(Create(y_axis), Write(y_label), run_time=2)
        self.play(Write(line_at_level_1), Write(y_equals_1_label), run_time=3.5)
        self.wait(2) # 22s
        self.play(*(ones_anims + ones_anims_names), run_time=1.5)
        self.wait(0.33)
        self.play(*turn_to_ones_anims, run_time=1.5)
        self.wait(0.33)
        self.play(*turn_to_zeros_anims, run_time=1.5)
        self.wait(0.33)
        self.wait(0.5)
        self.play(*destroy_anims, run_time=1) # 28s

        self.play(Create(linear_fcn), run_time=1.5)
        self.play(Transform(straight_line_to_transform, fitted_straight_line), run_time=1.5)
        self.wait(0.1)

        ################################################################################################ Part 2
        stud_shifts = [new_pos - lab.get_center() for new_pos, lab in zip(stud_positions_on_line, result_labels)]
        stud_linear_shift_anims = [res.animate.shift(shift) for res, shift in zip(result_labels, stud_shifts)]
        stud_name_shift_anims = [res.animate.shift(shift) for res, shift in zip(student_names, stud_shifts)]
        self.play(*(stud_linear_shift_anims + stud_name_shift_anims), run_time=1.4) # 32.5s

        logistic_fcn, logistic_fcn_dict = get_func_graph(a=0.6, b=0.5, x_range=[-8, 8],
                                                         y_range=[-0.5, 1.2],
                                                         x_length=graph_x_length,
                                                         y_length=graph_y_length, scale=graph_scale,
                                                         type='sigmoid')
        logistic_fcn.move_to(np.array([4.5, 2.5, 0]))
        equation = MathTex(r'z = \frac{1}{1 + e^{-y}}', font_size=24, color=YELLOW).next_to(logistic_fcn_dict["ax"],
                                                                                            RIGHT * 0.1). \
            shift(UP * 0.9 + LEFT * 0.5)
        axis_labels = logistic_fcn_dict["ax"].get_axis_labels(x_label='y', y_label='z')
        logistic_fcn_dict["fcn"] = give_rainbow_colors(logistic_fcn_dict["fcn"])
        label_scale = 0.75
        axis_labels[0].scale(label_scale)
        axis_labels[1].scale(label_scale).shift(LEFT * 0.5)
        logistic_fcn.add(equation)
        logistic_fcn.add(axis_labels)
        logistic_fcn.shift(DOWN * v_shift)
        self.play(Create(logistic_fcn), run_time=1.4)
        logistic_ghost = get_ghost(logistic_fcn, opacity=0.6)

        def scale_down(mobj, dt):
            mobj.scale(1 - dt * 0.4)

        logistic_ghost.add_updater(scale_down)
        self.play(logistic_ghost.animate.move_to(straight_line_to_transform.get_center()), run_time=1.4)
        self.remove(logistic_ghost)

        def logistic_func_big(t):
            return 1 / (1 + np.exp(-(1.3 * t - x_range[1] / 2)))

        logistic_curve_big = ParametricFunction(lambda t: hours_studied_graph_dict["original_coords"].
                                                c2p(t, logistic_func_big(t)), t_range=x_range)
        logistic_curve_big = give_rainbow_colors(logistic_curve_big)
        self.play(Transform(straight_line_to_transform, logistic_curve_big), run_time=1.4) # 37s

        results_on_line_y = [logistic_func_big(x) for x in study_hours]
        stud_positions_on_line = [hours_studied_graph_dict["original_coords"].
                                  c2p(x, y) for x, y in zip(study_hours, results_on_line_y)]
        stud_shifts = [new_pos - lab.get_center() for new_pos, lab in zip(stud_positions_on_line, result_labels)]
        stud_linear_shift_anims = [res.animate.shift(shift) for res, shift in zip(result_labels, stud_shifts)]
        stud_name_shift_anims = [res.animate.shift(shift) for res, shift in zip(student_names, stud_shifts)]
        self.wait(4)
        self.play(*(stud_linear_shift_anims + stud_name_shift_anims), run_time=2)
        # 44s

        line_start = hours_studied_graph.get_left() + hours_studied_graph_dict["unit_vec_y"] / 2
        line_end = hours_studied_graph.get_right() + hours_studied_graph_dict["unit_vec_y"] / 2
        line_at_level_05 = DashedLine(start=line_start, end=line_end, color=YELLOW, stroke_width=stroke_width_line05)
        y_equals_05_label = Tex(r'y = 0.5', font_size=35, color=YELLOW).next_to(line_at_level_05, RIGHT)

        label_0 = get_student_label(result_labels[0], 0, numeric=False)
        label_1 = get_student_label(result_labels[-1], 1, numeric=False)

        unknown_labels = [label for label, label_num, result in zip(result_labels,
                                                                    result_labels_numeric,
                                                                    results) if result is None]
        pos0 = hours_studied_graph_dict["original_coords"].c2p(study_hours[1], logistic_func_big(study_hours[1]))
        pos1 = hours_studied_graph_dict["original_coords"].c2p(study_hours[6], logistic_func_big(study_hours[6]))
        turn_to_zeros_anims = [Transform(unknown_labels[0], label_0.move_to(pos0))]
        turn_to_ones_anims = [Transform(unknown_labels[1], label_1.move_to(pos1))]
        r = 0.24
        self.play(Write(line_at_level_05), Write(y_equals_05_label), run_time=2)
        self.play(Write(self.get_student_circle(radius=r).move_to(pos1)), run_time=1)
        self.play(*turn_to_ones_anims, run_time=1.5)
        self.wait(1) # 50.5s
        self.play(Write(self.get_student_circle(radius=r).move_to(pos0)), run_time=1.5)
        self.wait(0.5)
        self.play(*turn_to_zeros_anims, run_time=1.5) # 54s

        y_100_percent = Tex(r'100\%', font_size=35, color=YELLOW).next_to(line_at_level_1, RIGHT). \
            move_to(y_equals_1_label.get_center())
        y_50_percent = Tex(r'50\%', font_size=35, color=YELLOW).next_to(line_at_level_1, RIGHT). \
            move_to(y_equals_05_label.get_center())
        self.wait(10.5)
        self.play(Transform(y_equals_05_label, y_50_percent), run_time=1.5)
        # 1:06s
        self.wait(2)
        self.play(Transform(y_equals_1_label, y_100_percent), run_time=1.5) # 1:09.5
        self.wait(1.5)

        self.play(Uncreate(logistic_fcn), Uncreate(linear_fcn), run_time=1)
        # 1:12s

        traffic_light_time = 0.2

        def shift_colors(shift):
            curve = give_rainbow_colors(straight_line_to_transform.copy(), shift=shift)
            self.play(Transform(straight_line_to_transform, curve), run_time=traffic_light_time)

        for i in range(15):
            shift_colors(i)

        cost_fcn_reveal_time = 3.5
        cost_function = MathTex(r'L(a, b)', r'&=',
                                r'- \frac{1}{m} \sum_{i=1}^{m}\left[y_i \log(h(x_i)) + (1 - y_i) \log(1 - h(x_i))\right]\;,\\',
                                r'h(x_i)', r'&=', r'\frac{1}{1 + e^{-(a x_i + b)}}',
                                font_size=35, color=YELLOW).shift(UP * 2.4)
        n_anims = int(cost_fcn_reveal_time / traffic_light_time)

        def get_shifted_colors(shift):
            return give_rainbow_colors(straight_line_to_transform.copy(), shift=shift)

        rainbow_anims = []
        for i in range(n_anims):
            new_curve = get_shifted_colors(i + 1)
            rainbow_anims.append(Transform(straight_line_to_transform, new_curve, run_time=traffic_light_time))

        anims_ = Succession(*rainbow_anims)

        self.wait(1.5)
        self.play(Write(cost_function, run_time=cost_fcn_reveal_time), anims_) # total runtime = 3.5s
        # 1:17s --> actually 1:19.5s

        all_mobjects = Group()
        for mobj in self.mobjects:
            all_mobjects.add(mobj)

        all_mobjects_small = all_mobjects.copy().scale(0.6).shift(UP)
        self.wait(14)
        self.play(Transform(all_mobjects, all_mobjects_small), run_time=1.5) # 1:35s

        question_font = 55
        question1 = Tex(r'\begin{minipage}{15cm}1. Why logistic function?\end{minipage}', font_size=question_font)
        question2 = Tex(r'\begin{minipage}{15cm}2. But how do I solve this?\end{minipage}', font_size=question_font)
        ques = VGroup(question1, question2)
        ques.arrange(DOWN, center=False, aligned_edge=LEFT).shift(DOWN * 2)
        self.wait(3)
        self.play(Write(question1), run_time=2.5) # 1:39.5s
        self.wait(5)
        self.play(Write(question2), run_time=2.5) # 1:45.5s
        self.wait(8.45 - 1.5)
        self.play(FadeOut(all_mobjects), Unwrite(question1), Unwrite(question2), run_time=2)
        self.wait(0.5)
        # 1:56s


class LinearRegression(ThreeDScene):
    def construct(self):
        config.disable_caching = True

        ########################################## Axes
        x_range = [-4.5, 4.5]
        y_range = [-3, 3]
        x_length = 8
        y_length = 5
        axes = get_axes(x_range=x_range,
                        y_range=y_range,
                        x_length=x_length,
                        y_length=y_length,
                        scale=1.2,
                        arrow_tip_scale=0.2).scale(0.9).shift(DOWN * 0.7)
        axis_labels = axes.get_axis_labels(x_label='x', y_label='y').set_color(WHITE)
        self.play(Create(axes), Write(axis_labels), run_time=1.5)

        ########################################## Linear regression
        N = 35
        np.random.seed(43)
        x_data = (np.random.random(N) - 0.5) * x_range[1] * 2
        x_data = np.sort(x_data)
        a = 0.7
        # b = 0.5
        b = 1
        y_data = x_data * a + b + np.random.randn(N) * 0.7
        x_data = list(x_data)
        y_data = list(y_data)
        res = stats.linregress(np.array(x_data), np.array(y_data))
        fitted_straight_line = ParametricFunction(lambda t: axes.
                                                  c2p(t, res.slope * t + res.intercept), t_range=x_range)
        fitted_straight_line = give_rainbow_colors(fitted_straight_line)
        dots = [Circle(radius=0.03, fill_color=RED, fill_opacity=1).move_to(axes.c2p(x, y))
                for x, y in zip(x_data, y_data)]
        point_creation_time = 1
        time_to_create_all_points = 4.8
        start_times = np.random.random(len(dots)) * (time_to_create_all_points - point_creation_time)
        anims = [Succession(Wait(run_time=st), FadeIn(p, run_time=point_creation_time),
                            Wait(run_time=time_to_create_all_points - point_creation_time - st))
                 for p, st in zip(dots, start_times)]
        self.play(*anims) # 4s in total
        self.play(Write(fitted_straight_line), run_time=2)

        font_size = 50
        equation = MathTex(r'y = ax + b', font_size=font_size, color=YELLOW).next_to(axes, UP * 3.5)
        self.play(Write(equation), run_time=2)

        ########################################## Params
        param_def = MathTex(r'\text{Parameters}&\hphantom{:}\\',
                            r'\text{Slope}', r'&:', r'a', r' = \tan(\alpha)\\',
                            r'\text{Intercept}', r'&:', r'b\\',
                            font_size=35, color=YELLOW).shift(LEFT * 4.8)
        param_def[0].set_color(WHITE)
        param_def[1].set_color(WHITE)
        param_def[2].set_color(WHITE)
        param_def[5].set_color(WHITE)
        param_def[6].set_color(WHITE)
        align_top(param_def, equation)
        self.wait(1.2)
        self.play(Write(param_def[0]), run_time=2) # 13s

        # Slope
        self.wait(1.5)
        self.play(Write(param_def[1]), Write(param_def[2]), Write(param_def[3]), run_time=2) # 16.5s
        angle = np.arctan(a)
        r = 2.4
        delta0 = 0.008
        delta1 = 0.025
        e = ValueTracker(0)

        def get_coords(t):
            center = [-res.intercept / res.slope, 0]
            new_coords = [center[0] + r * np.cos(t), center[1] + r * np.sin(t)]
            return axes.c2p(new_coords[0], new_coords[1])

        arc = always_redraw(lambda: ParametricFunction(lambda t: get_coords(t),
                                                       t_range=[angle - e.get_value() - delta1, angle - delta1],
                                                       color=GREEN))
        self.add(arc)
        self.play(e.animate.set_value(angle - (delta0 + delta1)), run_time=1.5)
        slope_label = MathTex(r'\alpha', font_size=font_size, color=GREEN).next_to(arc, RIGHT). \
            shift(UP * 0.15 + LEFT * 0.15)
        self.play(Write(slope_label), run_time=1.5)
        self.play(Write(param_def[4]), run_time=1.5)
        self.wait(0.5)
        # 21.5s


        # Intercept
        self.wait(1)
        self.play(Write(param_def[5]), Write(param_def[6]), Write(param_def[7]), run_time=2) # 24.5
        brace_template = BraceBetweenPoints(axes.c2p(0, 0), axes.c2p(0, res.intercept)).set_color(GREEN). \
            shift(LEFT * 0.15)
        brace_shrink_factor = 2
        brace = BraceBetweenPoints(axes.c2p(0, 0), axes.c2p(0, res.intercept * brace_shrink_factor)).set_color(GREEN). \
            shift(LEFT * 0.15).scale(1 / brace_shrink_factor).set_opacity(1)
        align_left(brace, brace_template)
        align_bottom(brace, brace_template)
        b = MathTex(r'b', font_size=35).set_color(YELLOW).next_to(brace, RIGHT * 0.5)
        self.wait(1.5)
        self.play(Write(brace), Write(b), run_time=1.5)
        self.wait(0.5)
        # 28s

        ######################################### Clear the screen
        self.wait(5.3)
        all_mobjects = []
        for d in self.mobjects:
            all_mobjects.append(d)
        self.play(*[FadeOut(m) for m in all_mobjects], run_time=1.5)
        self.wait(0.5)


class WhyLogRes(ThreeDScene):

    def get_student_circle(self, radius=0.2):
        return Circle(radius=radius, color=WHITE, fill_opacity=0, stroke_opacity=1, stroke_width=2.5)

    def get_student_name(self, stud, name):
        return Tex(r'\begin{minipage}{15cm}' + name + r'\end{minipage}', font_size=22).next_to(stud, UP * 0.5). \
            set_z_index(1)

    def construct(self):
        scale = 1.2
        v_shift = 0.4
        x_range = [0, 9.8]
        y_range = [0, 1.75]
        x_length = 5.5 * scale
        y_length = 2 * scale
        hours_studied_graph, hours_studied_vgroup, hours_studied_graph_dict = get_coordinate_line(scene_obj=self,
                                                                                                  x_range=x_range,
                                                                                                  y_range=y_range,
                                                                                                  x_length=x_length,
                                                                                                  y_length=y_length,
                                                                                                  scale=1.2,
                                                                                                  arrow_tip_scale=0.2)
        x_label = Tex(r'\begin{minipage}{15cm}Hours of study\end{minipage}', font_size=35). \
            next_to(hours_studied_graph, DOWN).shift(RIGHT * 2)
        linear_formula = MathTex('y = ax + b', font_size=50, color=YELLOW).next_to(hours_studied_graph, UP * 10)
        hours_studied_vgroup.add(linear_formula)
        y_axis = hours_studied_graph_dict["y_coord_line"]
        y_label = Tex(r'y', font_size=35).next_to(y_axis, LEFT * 0.3).shift(UP * 1.5)
        all_parts_of_graph = VGroup(hours_studied_vgroup, y_axis, x_label, y_label)
        study_hours = [0.05, 0.5, 0.9, 1.4, 1.8, 2.3, 3.9, 4.3, 5, 7.5]
        main_graph_scale = 0.89
        all_parts_of_graph.shift(DOWN + LEFT * 2.1).scale(main_graph_scale)
        # student_circles = [self.get_student_circle().move_to(hours_studied_graph_dict["original_coords"].
        #                                                      c2p(hour, 0, 0)).shift(DOWN) for hour in study_hours]
        student_circles = [self.get_student_circle().move_to(hours_studied_graph_dict["original_coords"].
                                                             c2p(hour, 0, 0)) for hour in study_hours]

        ax = hours_studied_graph_dict["original_coords"]
        line_start = ax.c2p(0, 1, 0)
        line_end = ax.c2p(x_range[1], 1, 0)
        line_at_level_1 = DashedLine(start=line_start, end=line_end, color=YELLOW,
                                         stroke_width=stroke_width_line05)
        line_start = ax.c2p(0, 0.5, 0)
        line_end = ax.c2p(x_range[1], 0.5, 0)
        line_at_level_05 = DashedLine(start=line_start, end=line_end, color=YELLOW,
                                          stroke_width=stroke_width_line05)

        font_labels = 28
        y_equals_1_label = Tex(r'y = 1', font_size=font_labels, color=YELLOW).next_to(line_at_level_1, LEFT)
        y_equals_05_label = Tex(r'y = 0.5', font_size=font_labels, color=YELLOW).next_to(line_at_level_05, LEFT)

        results = [0, None, 0, 0, 0, 0, 1, 1, None, 1]
        student_circles_vgroup = VGroup(*student_circles)
        all_parts_of_graph.add(student_circles_vgroup)
        # study_hours_anims = [Create(stud) for stud in student_circles]
        result_font = 33
        result_labels = [get_student_label(stud, result, font_size=result_font)
                         for stud, result in zip(student_circles, results)]
        result_labels_numeric = [get_student_label(stud, result, numeric=True, font_size=result_font)
                                 for stud, result in zip(student_circles, results)]
        result_labels_anims = [Write(lab) for lab in result_labels]
        results_known = [result_labels_anims[i] for i in range(len(results)) if results[i] is not None]
        results_unknown = [result_labels_anims[i] for i in range(len(results)) if results[i] is None]
        stud_circle_removals = [Uncreate(stud) for stud in student_circles]
        stud_circle_removals_known = [stud_circle_removals[i] for i in range(len(results)) if results[i] is not None]
        stud_circle_removals_unknown = [stud_circle_removals[i] for i in range(len(results)) if results[i] is None]
        vert_unit_vec = hours_studied_graph_dict["original_coords"].c2p(0, 1, 0) - \
                        hours_studied_graph_dict["original_coords"].c2p(0, 0, 0)
        ones_anims = [label.animate.shift(vert_unit_vec) for label, result in zip(result_labels, results)
                      if result == 1]
        for label, result in zip(result_labels, results):
            if result == 1:
                label.shift(vert_unit_vec)
        for i in range(len(results)):
            if results[i] == 1:
                result_labels_numeric[i].shift(vert_unit_vec)
        turn_to_ones_anims = [Transform(label, label_num) for label, label_num, result in zip(result_labels,
                                                                                              result_labels_numeric,
                                                                                              results) if result == 1]
        turn_to_zeros_anims = [Transform(label, label_num) for label, label_num, result in zip(result_labels,
                                                                                               result_labels_numeric,
                                                                                               results) if result == 0]

        graph_x_range = [-4, 4]
        graph_y_range = [-3, 3]
        graph_x_length = 2.8
        graph_y_length = 1.9
        graph_scale = 1
        linear_fcn, linear_fcn_dict = get_func_graph(a=0.6, b=0.5, x_range=graph_x_range, y_range=graph_y_range,
                                                     x_length=graph_x_length,
                                                     y_length=graph_y_length, scale=graph_scale)
        linear_fcn.move_to(np.array([-4.5, 2.5, 0])).shift(DOWN * v_shift)
        equation = MathTex(r'y = ax + b', font_size=24, color=YELLOW).next_to(linear_fcn_dict["ax"], RIGHT * 0.1). \
            shift(UP * 1 + LEFT * 0.5)
        axis_labels = linear_fcn_dict["ax"].get_axis_labels(x_label='x', y_label='y')
        label_scale = 0.75
        axis_labels[0].scale(label_scale)
        axis_labels[1].scale(label_scale).shift(LEFT * 0.5)
        linear_fcn.add(equation)
        linear_fcn.add(axis_labels)
        x_data = []
        y_data = []
        for result, hour in zip(results, study_hours):
            if result is not None:
                x_data.append(hour)
                y_data.append(result)

        res = stats.linregress(np.array(x_data), np.array(y_data))
        fitted_straight_line = ParametricFunction(lambda t: hours_studied_graph_dict["original_coords"].
                                                  c2p(t, res.slope * t + res.intercept), t_range=x_range)
        fitted_straight_line = give_rainbow_colors(fitted_straight_line)
        linear_fcn_dict["fcn"] = give_rainbow_colors(linear_fcn_dict["fcn"])
        straight_line_to_transform = linear_fcn_dict["fcn"].copy()
        straight_line_to_transform.set_opacity(0.5)
        straight_line_to_transform = give_rainbow_colors(straight_line_to_transform)
        results_on_line_y = [res.slope * x + res.intercept for x in study_hours]
        stud_positions_on_line = [hours_studied_graph_dict["original_coords"].
                                  c2p(x, y) for x, y in zip(study_hours, results_on_line_y)]

        ################################################################################################

        self.wait(0.5)
        self.play(*(results_known + results_unknown + [Create(hours_studied_graph), Write(x_label),
                                                                           Create(y_axis), Write(y_label)]),
                  Write(line_at_level_1), Write(y_equals_1_label), Write(line_at_level_05), Write(y_equals_05_label), run_time=2.5)
        self.wait(4) #7s
        self.wait(0.5)
        self.play(Create(fitted_straight_line), Write(linear_formula), run_time=2)

        ################################################ Params
        param_v_shift = 0.4
        param_def = MathTex(r'\text{Parameters}&\hphantom{:}\\',
                            r'\text{Slope}', r'&:', r'a', r' = ' + str("{:.1f}".format(res.slope)) + r'\\',
                            r'\text{Intercept}', r'&:', r'b', r' = ' + str("{:.1f}".format(res.intercept)),
                            font_size=40, color=YELLOW).shift(LEFT * 4.8 + UP * 3).shift(DOWN * param_v_shift)
        param_def[0].set_color(WHITE)
        param_def[1].set_color(WHITE)
        param_def[2].set_color(WHITE)
        param_def[5].set_color(WHITE)
        param_def[6].set_color(WHITE)
        self.play(Write(param_def), run_time=1.5) #10s

        ################################################## Move to the line & classify
        shift_anims = [result.animate.move_to(hours_studied_graph_dict["original_coords"].
                                              c2p(hour, res.slope * hour + res.intercept))
         for result, hour in zip(result_labels, study_hours)]
        self.wait(1.5)
        self.play(*shift_anims, run_time=1.5) #13s

        ax = hours_studied_graph_dict["original_coords"]
        length = x_range[1]
        height = 1.35
        x_threshold = (0.5 - res.intercept) / res.slope
        points = {"A": list(ax.c2p(x_threshold, 0.5)),
                  "B": list(ax.c2p(x_threshold, height)),
                  "C": list(ax.c2p(length, height)),
                  "D": list(ax.c2p(length, 0.5))}
        pts = [val for key, val in points.items()]
        ones_rect = Polygon(*pts, color=GREEN, fill_opacity=0.1)

        points = {"A": list(ax.c2p(x_threshold, 0.5)),
                  "B": list(ax.c2p(x_threshold, 1 - height)),
                  "C": list(ax.c2p(-0.24, 1 - height)),
                  "D": list(ax.c2p(-0.24, 0.5))}
        pts = [val for key, val in points.items()]
        zeros_rect = Polygon(*pts, color=RED, fill_opacity=0.1)

        fail = None
        old_fail = None
        success = None
        old_success = None
        for stud, result, old_label in zip(student_circles, results, result_labels):
            if result is None:
                if fail is None:
                    fail = get_student_label(stud, 0, numeric=True, font_size=result_font)
                    fail_sign = get_student_label(stud, 0, numeric=False)
                    old_fail = old_label
                    fail = fail.move_to(old_fail)
                    fail_sign = fail_sign.move_to(old_fail)
                elif success is None:
                    success = get_student_label(stud, 1, numeric=True, font_size=result_font)
                    success_sign = get_student_label(stud, 1, numeric=False, font_size=result_font)
                    old_success = old_label
                    success = success.move_to(old_success)
                    success_sign = success_sign.move_to(old_success)
                    break

        self.wait(2.5)
        self.play(FadeIn(ones_rect), Transform(old_success, success), run_time=1.5) #17s
        self.wait(0.5)
        self.play(FadeIn(zeros_rect), Transform(old_fail, fail), run_time=1.5) #19s

        ################################################## Clean up, move points to 0, 1 lines, extend x-axis
        positions_01 = [ax.c2p(hours, 1, 0) if res.slope * hours + res.intercept > 0.5 else ax.c2p(hours, 0, 0)
                        for lab, hours in zip(result_labels, study_hours)]
        fail_template = result_labels[0].copy()
        success_template = result_labels[-1].copy()
        labels_new = [success_template.copy().move_to(pos) if res.slope * hours + res.intercept > 0.5 else
                      fail_template.copy().move_to(pos)
                      for lab, pos, hours in zip(result_labels, positions_01, study_hours)]
        anims = [Transform(lab, lab_new) for lab, lab_new in zip(result_labels, labels_new)]
        self.wait(1.5)
        self.play(*([FadeOut(ones_rect), FadeOut(zeros_rect), Transform(old_success, success_sign),
                  Transform(old_fail, fail_sign)] + anims), run_time=1.5) #22s

        new_x_max = 16.2
        new_x_length = x_length * new_x_max / x_range[1]
        graph_extended_x, hours_studied_vgroup, hours_studied_graph_dict = get_coordinate_line(scene_obj=self,
                                                                                                  x_range=[0, new_x_max],
                                                                                                  y_range=y_range,
                                                                                                  x_length=new_x_length,
                                                                                                  y_length=y_length,
                                                                                                  scale=1.2,
                                                                                                  arrow_tip_scale=0.2)
        graph_extended_x.scale(main_graph_scale)
        align_left(graph_extended_x, hours_studied_graph)
        align_bottom(graph_extended_x, hours_studied_graph)

        line_start_new = ax.c2p(0, 1, 0)
        line_end_new = ax.c2p(new_x_max, 1, 0)
        line_at_level_1_new = DashedLine(start=line_start_new, end=line_end_new, color=YELLOW,
                                         stroke_width=stroke_width_line05)
        line_start_new = ax.c2p(0, 0.5, 0)
        line_end_new = ax.c2p(new_x_max, 0.5, 0)
        line_at_level_05_new = DashedLine(start=line_start_new, end=line_end_new, color=YELLOW,
                                          stroke_width=stroke_width_line05)

        fitted_straight_line_new = ParametricFunction(lambda t: ax.c2p(t, res.slope * t + res.intercept),
                                                      t_range=[0, new_x_max])
        fitted_straight_line_new = give_rainbow_colors(fitted_straight_line_new)

        self.wait(1)
        self.play(Transform(hours_studied_graph, graph_extended_x), Transform(line_at_level_1, line_at_level_1_new),
                  Transform(line_at_level_05, line_at_level_05_new),
                  Transform(fitted_straight_line, fitted_straight_line_new), run_time=2.5)

        ################################################## Add more examples
        study_hours_extended = [12.9, 13.3, 13.7, 14.1, 14.5, 14.85, 15.4, 15.8, 16.2]
        results_extended = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        result_labels_extended = [success_template.copy().move_to(ax.c2p(hours, result, 0))
                                  for hours, result in zip(study_hours_extended, results_extended)]
        self.wait(0.5)
        self.play(*[Write(res) for res in result_labels_extended], run_time=2) #28s

        ################################################## Refit the line & change a and b
        x_all = x_data + study_hours_extended
        y_all = y_data + results_extended
        res_new = stats.linregress(np.array(x_all), np.array(y_all))
        line_adjusted = ParametricFunction(lambda t: ax.c2p(t, res_new.slope * t + res_new.intercept),
                                                      t_range=[0, new_x_max])
        line_adjusted = give_rainbow_colors(line_adjusted)

        param_new = MathTex(r'\text{Parameters}&\hphantom{:}\\',
                            r'\text{Slope}', r'&:', r'a', r' = ' + str("{:.1f}".format(res_new.slope)) + r'\\',
                            r'\text{Intercept}', r'&:', r'b', r' = ' + str("{:.1f}".format(res_new.intercept)),
                            font_size=40, color=YELLOW).shift(LEFT * 4.8 + UP * 3).shift(DOWN * param_v_shift)
        param_new[0].set_color(WHITE)
        param_new[1].set_color(WHITE)
        param_new[2].set_color(WHITE)
        param_new[5].set_color(WHITE)
        param_new[6].set_color(WHITE)
        align_left(param_new, param_def)

        self.wait(0.5)
        self.play(Transform(fitted_straight_line, line_adjusted), Transform(param_def, param_new), run_time=2)
        self.wait(2)

        ################################################## Change labels
        all_labels = result_labels + result_labels_extended
        new_labels = [lab.copy() for lab in all_labels]
        start_miscalssified = 6
        end_misclassified = 9
        for i in range(start_miscalssified, end_misclassified):
            new_labels[i] = fail_template.copy().move_to(new_labels[i].get_center()).set_color(PURPLE_A)
        anims = [Transform(lab, lab_new) for lab, lab_new in zip(all_labels, new_labels)]
        self.play(*anims, run_time=1.5)

        # Rectangular box
        misclassified = VGroup(*(all_labels[start_miscalssified:end_misclassified]))
        surround_rect = Rectangle(height=0.5, width=1.2, color=WHITE).move_to(misclassified.get_center())
        self.play(Create(surround_rect), run_time=1.5) #35.5s
        self.wait(4.5) #40s

        ################################################## Remove the extra points & refit the line
        point_remove_anims = [Unwrite(p) for p in result_labels_extended]
        anims_labels = []
        for i in range(start_miscalssified, end_misclassified):
            lab = success_template.copy().move_to(all_labels[i].get_center())
            anims_labels.append(Transform(all_labels[i], lab))

        line_adjusted = ParametricFunction(lambda t: ax.c2p(t, res.slope * t + res.intercept),
                                                      t_range=[0, new_x_max])
        line_adjusted = give_rainbow_colors(line_adjusted)

        param_new = MathTex(r'\text{Parameters}&\hphantom{:}\\',
                            r'\text{Slope}', r'&:', r'a', r' = ' + str("{:.1f}".format(res.slope)) + r'\\',
                            r'\text{Intercept}', r'&:', r'b', r' = ' + str("{:.1f}".format(res.intercept)),
                            font_size=40, color=YELLOW).shift(LEFT * 4.8 + UP * 3).shift(DOWN * param_v_shift)
        param_new[0].set_color(WHITE)
        param_new[1].set_color(WHITE)
        param_new[2].set_color(WHITE)
        param_new[5].set_color(WHITE)
        param_new[6].set_color(WHITE)
        align_left(param_new, param_def)

        self.wait(1)
        self.play(*(point_remove_anims + anims_labels + [Uncreate(surround_rect),
                                                         Transform(fitted_straight_line, line_adjusted),
                                                         Transform(param_def, param_new)]), run_time=2.5)
        self.wait(1)
        # 44.5s

        ################################################## Animate the distance between a point and the line
        moving_point_hours = ValueTracker(9)
        moving_point_y = ValueTracker(1)
        moving_point_template = success_template.copy()

        def get_moving_point():
            moving_point_template.move_to(ax.c2p(moving_point_hours.get_value(), moving_point_y.get_value(), 0))
            return moving_point_template

        def get_brace_point_line():
            delta_x = 0.2
            return BraceBetweenPoints(ax.c2p(moving_point_hours.get_value() - delta_x, 1),
                                      ax.c2p(moving_point_hours.get_value() - delta_x,
                                             res.slope * (moving_point_hours.get_value() - delta_x) + res.intercept)).\
                set_color(BLUE)

        moving_point = always_redraw(get_moving_point)
        moving_brace = always_redraw(get_brace_point_line)

        self.play(Write(moving_point), run_time=1)
        self.play(Write(moving_brace), run_time=1.5)
        self.play(moving_point_hours.animate.set_value(study_hours_extended[-1]), run_time=2)
        #49s

        ################################################## Add more examples
        self.wait(1)
        self.play(Unwrite(moving_brace), run_time=1)
        study_hours_remaining = study_hours_extended[:-1]
        results_remaining = results_extended[:-1]
        result_labels_remaining = [success_template.copy().move_to(ax.c2p(hours, result, 0))
                                  for hours, result in zip(study_hours_remaining, results_remaining)]
        self.play(*[Write(res) for res in result_labels_remaining], run_time=1)

        ################################################## Refit the line & change a and b
        res_new = stats.linregress(np.array(x_all), np.array(y_all))
        line_adjusted = ParametricFunction(lambda t: ax.c2p(t, res_new.slope * t + res_new.intercept),
                                                      t_range=[0, new_x_max])
        line_adjusted = give_rainbow_colors(line_adjusted)

        param_new = MathTex(r'\text{Parameters}&\hphantom{:}\\',
                            r'\text{Slope}', r'&:', r'a', r' = ' + str("{:.1f}".format(res_new.slope)) + r'\\',
                            r'\text{Intercept}', r'&:', r'b', r' = ' + str("{:.1f}".format(res_new.intercept)),
                            font_size=40, color=YELLOW).shift(LEFT * 4.8 + UP * 3).shift(DOWN * param_v_shift)
        param_new[0].set_color(WHITE)
        param_new[1].set_color(WHITE)
        param_new[2].set_color(WHITE)
        param_new[5].set_color(WHITE)
        param_new[6].set_color(WHITE)
        align_left(param_new, param_def)

        self.play(Transform(fitted_straight_line, line_adjusted), Transform(param_def, param_new), run_time=1.9)

        ################################################## Move points to the line
        all_x_pos = study_hours + study_hours_extended
        pos_on_line = [ax.c2p(x, res_new.slope * x + res_new.intercept) for x in all_x_pos]
        all_labels = result_labels + result_labels_remaining + [result_labels_extended[-1]]
        move_to_line_anims = [label.animate.move_to(pos) for label, pos in zip(all_labels, pos_on_line)]
        self.play(*(move_to_line_anims +
                    [moving_point_y.animate.set_value(res_new.slope * all_x_pos[-1] + res_new.intercept)]), run_time=1.25)

        ################################################## Change labels for misclassified points
        all_labels = result_labels + result_labels_extended
        new_labels = [lab.copy() for lab in all_labels]
        start_miscalssified = 6
        end_misclassified = 9
        for i in range(start_miscalssified, end_misclassified):
            new_labels[i] = fail_template.copy().move_to(new_labels[i].get_center()).set_color(PURPLE_A)
        anims = [Transform(lab, lab_new) for lab, lab_new in zip(all_labels, new_labels)]
        self.play(*anims, run_time=1.25)
        # 57s

        ################################################## Logistic Regression
        clf = LogisticRegression(random_state=0).fit(np.array(x_all).reshape(-1, 1), np.array(y_all))

        def logistic_func(t):
            return 1 / (1 + np.exp(-(clf.coef_ * t + clf.intercept_)))

        log_curve = give_rainbow_colors(ParametricFunction(lambda t:
                                                           ax.c2p(t, logistic_func(t)), t_range=[0, new_x_max]))

        pos_on_curve = [ax.c2p(x, logistic_func(x)) for x in all_x_pos]
        all_labels = result_labels + result_labels_remaining
        move_to_line_anims = [label.animate.move_to(pos) for label, pos in zip(all_labels, pos_on_curve)]
        move_to_line_anims.append(moving_point_y.animate.set_value(logistic_func(all_x_pos[-1])))

        logistic_equation = MathTex(r'\frac{1}{1 + e', r'^{-(', r'ax + b', r')}}', font_size=50, color=YELLOW).\
            move_to(linear_formula)

        self.wait(1.5)
        self.play(*([Transform(fitted_straight_line, log_curve), Transform(linear_formula, logistic_equation)] +
                    move_to_line_anims), run_time=2)

        ################################################## Correctly classify points
        new_labels = [lab.copy() for lab in all_labels]
        start_miscalssified = 6
        end_misclassified = 9
        for i in range(start_miscalssified, end_misclassified):
            new_labels[i] = success_template.copy().move_to(new_labels[i].get_center())
        anims = [Transform(lab, lab_new) for lab, lab_new in zip(all_labels, new_labels)]
        self.wait(0.5)
        self.play(*anims, run_time=2)
        # 1:03s

        ################################################## Add more examples to the right
        new_x_right = [9, 10, 10.7, 11.3]
        new_y_right = [logistic_func(x) for x in new_x_right]
        new_pos_right = [ax.c2p(x, y) for x, y in zip(new_x_right, new_y_right)]
        new_labels_right = [success_template.copy().move_to(pos) for pos in new_pos_right]
        self.wait(1)
        self.play(*[Write(lab) for lab in new_labels_right], run_time=1.5) # 1:05.5s

        ################################################## Add more examples to the left
        new_x_left = [2.8, 3.2]
        new_y_left = [logistic_func(x) for x in new_x_left]
        new_pos_left = [ax.c2p(x, y) for x, y in zip(new_x_left, new_y_left)]
        new_labels_left = [fail_template.copy().move_to(pos) for pos in new_pos_left]
        self.wait(0.5)
        self.play(*[Write(lab) for lab in new_labels_left], run_time=1.5) #1:07.5s

        ################################################## Refit the line & change a and b
        clf = LogisticRegression(random_state=0).\
            fit(np.array(x_all + new_x_right + new_x_left).reshape(-1, 1), np.array(y_all + [1, 1, 1, 1] + [0, 0]))

        def logistic_func(t):
            return 1 / (1 + np.exp(-(clf.coef_ * t + clf.intercept_)))

        log_curve = give_rainbow_colors(ParametricFunction(lambda t:
                                                           ax.c2p(t, logistic_func(t)), t_range=[0, new_x_max]))

        param_new = MathTex(r'\text{Parameters}&\hphantom{:}\\',
                            r'\text{Slope}', r'&:', r'a', r' = ' + str("{:.1f}".format(res_new.slope)) + r'\\',
                            r'\text{Intercept}', r'&:', r'b', r' = ' + str("{:.1f}".format(res_new.intercept)),
                            font_size=40, color=YELLOW).shift(LEFT * 4.8 + UP * 3).shift(DOWN * param_v_shift)
        param_new[0].set_color(WHITE)
        param_new[1].set_color(WHITE)
        param_new[2].set_color(WHITE)
        param_new[5].set_color(WHITE)
        param_new[6].set_color(WHITE)
        align_left(param_new, param_def)

        ################################################## Move points to the line
        pos_on_curve = [ax.c2p(x, logistic_func(x)) for x in all_x_pos]
        all_labels = result_labels + result_labels_remaining
        move_to_line_anims = [label.animate.move_to(pos) for label, pos in zip(all_labels, pos_on_curve)]
        move_to_line_anims.append(moving_point_y.animate.set_value(logistic_func(all_x_pos[-1])))

        pos_on_curve_right = [ax.c2p(x, logistic_func(x)) for x in new_x_right]
        move_to_line_anims_right = [label.animate.move_to(pos)
                                    for label, pos in zip(new_labels_right, pos_on_curve_right)]

        pos_on_curve_left = [ax.c2p(x, logistic_func(x)) for x in new_x_left]
        move_to_line_anims_left = [label.animate.move_to(pos)
                                    for label, pos in zip(new_labels_left, pos_on_curve_left)]

        self.wait(3)
        self.play(*([Transform(fitted_straight_line, log_curve), Transform(param_def, param_new)] +
                    move_to_line_anims + move_to_line_anims_right + move_to_line_anims_left), run_time=2.5)
        # 1: 13s

        self.wait(3)
        self.play(*[FadeOut(mobj) for mobj in self.mobjects], run_time=1.5)
        self.wait(0.5)
        # 1:18s


class Outro(ThreeDScene):

    def get_student_circle(self, radius=0.2):
        return Circle(radius=radius, color=WHITE, fill_opacity=0, stroke_opacity=1, stroke_width=2.5)

    def get_student_name(self, stud, name):
        return Tex(r'\begin{minipage}{15cm}' + name + r'\end{minipage}', font_size=22).next_to(stud, UP * 0.5). \
            set_z_index(1)

    def construct(self):
        ################################ Params
        v_shift = 0.4
        study_hours = [1.3, 2.5, 3.7, 5, 6, 7, 8.5]
        results = [0, 0, 0, 1, 1, 1, 1]

        x_data = study_hours
        y_data = results
        res = stats.linregress(np.array(x_data), np.array(y_data))

        param_def = MathTex(r'\text{Parameters}&\hphantom{:}\\',
                            r'\text{Slope}', r'&:', r'a', r' = ' + str("{:.1f}".format(res.slope)) + r'\\',
                            r'\text{Intercept}', r'&:', r'b', r' = ' + str("{:.1f}".format(res.intercept)),
                            font_size=28, color=YELLOW).shift(LEFT * 4.8)
        param_def[0].set_color(WHITE)
        param_def[1].set_color(WHITE)
        param_def[2].set_color(WHITE)
        param_def[5].set_color(WHITE)
        param_def[6].set_color(WHITE)
        self.play(Write(param_def[0]), Write(param_def[1]), Write(param_def[2]), Write(param_def[3]),
                  Write(param_def[5]), Write(param_def[6]), Write(param_def[7]), run_time=2)
        self.wait(1) # 3s

        ################################ Small linear function
        graph_x_range = [-4, 4]
        graph_y_range = [-3, 3]
        graph_x_length = 2.8
        graph_y_length = 1.9
        graph_scale = 1
        linear_fcn, linear_fcn_dict = get_func_graph(a=res.slope, b=res.intercept, x_range=graph_x_range,
                                                     y_range=graph_y_range,
                                                     x_length=graph_x_length,
                                                     y_length=graph_y_length, scale=graph_scale)
        linear_fcn.move_to(np.array([-4.5, 2.5, 0])).shift(DOWN * v_shift)
        equation = MathTex(r'y = ', r'ax + b', font_size=24, color=YELLOW).next_to(linear_fcn_dict["ax"], RIGHT * 0.1). \
            shift(UP * 1 + LEFT * 0.5)
        axis_labels = linear_fcn_dict["ax"].get_axis_labels(x_label='x', y_label='y')
        label_scale = 0.75
        axis_labels[0].scale(label_scale)
        axis_labels[1].scale(label_scale).shift(LEFT * 0.5)
        linear_fcn.add(equation)
        linear_fcn.add(axis_labels)
        linear_fcn_dict["fcn"] = give_rainbow_colors(linear_fcn_dict["fcn"])
        straight_line_to_transform = linear_fcn_dict["fcn"].copy()
        straight_line_to_transform.set_opacity(0.5)
        straight_line_to_transform = give_rainbow_colors(straight_line_to_transform)
        linear_part_small = get_ghost(equation[1], opacity=1)
        self.wait(0.5)
        self.play(Create(linear_fcn), run_time=2) # 5.5s

        ################################ Assign values to a & b
        self.wait(2)
        self.play(Write(param_def[4]), Write(param_def[8]), run_time=1.5) #9s

        ################################ Logistic graph
        logistic_fcn, logistic_fcn_dict = get_func_graph(a=0.6, b=0.5, x_range=[-8, 8],
                                                         y_range=[-0.5, 1.2],
                                                         x_length=graph_x_length,
                                                         y_length=graph_y_length, scale=graph_scale,
                                                         type='sigmoid')
        logistic_fcn.move_to(np.array([4.5, 2.5, 0])).shift(DOWN * v_shift)
        equation = MathTex(r'z = ', r'\frac{1}{1 + e', r'^{-y}}', font_size=24, color=YELLOW).next_to(logistic_fcn_dict["ax"],
                                                                                            RIGHT * 0.1). \
            shift(UP * 0.9 + LEFT * 0.5)
        axis_labels = logistic_fcn_dict["ax"].get_axis_labels(x_label='y', y_label='z')
        logistic_fcn_dict["fcn"] = give_rainbow_colors(logistic_fcn_dict["fcn"])
        label_scale = 0.75
        axis_labels[0].scale(label_scale)
        axis_labels[1].scale(label_scale).shift(LEFT * 0.5)
        logistic_fcn.add(equation)
        logistic_fcn.add(axis_labels)
        self.play(Create(logistic_fcn), run_time=2) #11s

        ################################ Merge functions
        main_function = MathTex(r'\frac{1}{1 + e', r'^{-(', r'ax + b', r')}}',
                                font_size=40, color=YELLOW).shift(UP * 3)
        main_function.shift(DOWN * v_shift * 2)
        sigmoid_part = main_function[0]
        linear_part = main_function[2]
        sigmoid_part_small = get_ghost(equation[1], opacity=1)
        self.wait(0.5)
        self.play(Transform(sigmoid_part_small, sigmoid_part), Transform(linear_part_small, linear_part), run_time=2.4)
        self.play(Write(main_function[1]), Write(main_function[3]), run_time=0.6)
        # 14.5s

        ################################ Main graph
        x_range = [0, 9.8]
        y_range = [0, 1.75]
        x_length = 5.5
        y_length = 2
        hours_studied_graph, hours_studied_vgroup, hours_studied_graph_dict = get_coordinate_line(scene_obj=self,
                                                                                                  x_range=x_range,
                                                                                                  y_range=y_range,
                                                                                                  x_length=x_length,
                                                                                                  y_length=y_length,
                                                                                                  scale=1.2,
                                                                                                  arrow_tip_scale=0.2)
        x_label = Tex(r'\begin{minipage}{15cm}x\end{minipage}', font_size=35). \
            next_to(hours_studied_graph, DOWN).shift(RIGHT * 3.3)
        y_axis = hours_studied_graph_dict["y_coord_line"]
        y_label = Tex(r'y', font_size=35).next_to(y_axis, LEFT * 0.3).shift(UP * 0.2)
        all_parts_of_graph = VGroup(hours_studied_vgroup, y_axis, x_label)
        student_circles = [self.get_student_circle().move_to(hours_studied_graph_dict["original_coords"].
                                                             c2p(hour, 0, 0)).shift(DOWN) for hour in study_hours]
        all_parts_of_graph.shift(DOWN)
        student_circles = VGroup(*student_circles)
        all_parts_of_graph.add(student_circles)
        study_hours_anims = [Create(stud) for stud in student_circles]

        fitted_straight_line = ParametricFunction(lambda t: hours_studied_graph_dict["original_coords"].
                                                  c2p(t, res.slope * t + res.intercept), t_range=x_range)
        fitted_straight_line = give_rainbow_colors(fitted_straight_line)
        all_parts_of_graph.add(fitted_straight_line)
        all_parts_of_graph.add(y_label)

        def logistic_func_big(t):
            return 1 / (1 + np.exp(-(1.3 * t - x_range[1] / 2)))

        results_on_line_y = [logistic_func_big(x) for x in study_hours]
        result_labels = [get_student_label(stud, result) for stud, result in zip(student_circles, results)]
        for lab in result_labels:
            all_parts_of_graph.add(lab)
        all_parts_of_graph.scale(0.8)
        self.play(Create(hours_studied_graph), Write(x_label), Create(y_axis), Write(y_label), run_time=1)
        self.play(*study_hours_anims, run_time=1)
        self.play(Transform(straight_line_to_transform, fitted_straight_line), run_time=1)
        #17.5s

        logistic_ghost = get_ghost(logistic_fcn, opacity=0.6)

        def scale_down(mobj, dt):
            mobj.scale(1 - dt * 0.4)

        logistic_ghost.add_updater(scale_down)
        self.play(logistic_ghost.animate.move_to(straight_line_to_transform.get_center()), run_time=1.5)
        self.remove(logistic_ghost)

        logistic_curve_big = ParametricFunction(lambda t: hours_studied_graph_dict["original_coords"].
                                                c2p(t, logistic_func_big(t)), t_range=x_range)
        logistic_curve_big = give_rainbow_colors(logistic_curve_big)
        self.play(Transform(straight_line_to_transform, logistic_curve_big), run_time=1.5)

        line_start = hours_studied_graph.get_left() + hours_studied_graph_dict["unit_vec_y"] / 2
        line_end = hours_studied_graph.get_right() + hours_studied_graph_dict["unit_vec_y"] / 2

        line_at_level_05 = DashedLine(start=line_start, end=line_end, color=YELLOW, stroke_width=stroke_width_line05)
        y_equals_05_label = Tex(r'y = 0.5', font_size=35, color=YELLOW).next_to(line_at_level_05, RIGHT)

        stud_positions_on_line = [hours_studied_graph_dict["original_coords"].
                                  c2p(x, y) for x, y in zip(study_hours, results_on_line_y)]
        stud_shift_anims = [stud.animate.move_to(pos) for stud, pos in zip(student_circles, stud_positions_on_line)]
        self.play(*stud_shift_anims, run_time=1.5)

        self.play(Write(line_at_level_05), Write(y_equals_05_label), run_time=1.5)

        for lab, stud in zip(result_labels, student_circles):
            lab.move_to(stud.get_center()).scale(0.9)
        result_labels_anims = [Write(lab) for lab in result_labels]

        self.play(*result_labels_anims, run_time=1.5)
        #25s

        # Algorithm animation
        algo_block = Cube(side_length=1, stroke_color=PURPLE, stroke_opacity=1, stroke_width=2).\
            rotate(axis=UP, angle=10 * DEGREES).\
            next_to(hours_studied_graph, DOWN * 2)#.set_color(GREEN)

        self.wait(6)
        self.play(Create(algo_block), run_time=2) #33s

        N = 40
        data_font_size = 25

        self.opacity = 1

        input_x = [MathTex(r'x_{' + str(i + 1) + r'}', font_size=data_font_size).next_to(algo_block, LEFT * 3.5).
                   shift(LEFT * i * 0.9)
                   for i in range(N)]
        input_y = [MathTex(r'y_{' + str(i + 1) + r'}', font_size=data_font_size).next_to(input_x[i], DOWN * 0.8)
                   for i in range(N)]

        inputs = Group()
        data_anims = []
        all_data = Group()
        for x, y in zip(input_x, input_y):
            align_left(y, x)
            data_anims.append(Write(x))
            data_anims.append(Write(y))
            inputs.add(Group(x, y))
            all_data.add(x)
            all_data.add(y)
        for inp in inputs:
            align_vertically(inp, algo_block)

        e = ValueTracker(0)

        def fader(mobj, dt):
            mobj.move_to(mobj.init_pos + e.get_value() * RIGHT)
            if mobj.get_right()[0] > algo_block.get_left()[0] - 0.5:
                print("\nself.opacity = {:.4f}  x = {:.4f}".format(self.opacity, mobj.get_right()[0]))
                mobj.set_opacity(self.opacity)
                mobj.set_stroke_opacity(self.opacity)
                mobj.set_fill_opacity(self.opacity)
                self.opacity = max(0, self.opacity - 0.5 * dt)

        for x in input_x:
            x.init_pos = x.get_center()
            x.add_updater(fader)

        for y in input_y:
            y.init_pos = y.get_center()
            y.add_updater(fader)

        # Arrow
        width = 2
        tip_height = 0.8
        shaft_length = 5
        tip_length = 2.5
        arrow_tracker = ValueTracker(0)
        move_to = algo_block.get_center() + RIGHT * 1.2
        arrow = FilledArrowAnimated(arrow_tracker=arrow_tracker, width=width, tip_height=tip_height,
                                    shaft_length=shaft_length, tip_length=tip_length, scale=0.08, move_to=move_to,
                                    color=WHITE).get()

        self.add(arrow)

        font_size = data_font_size + 10
        final_a = MathTex(r'a', font_size=font_size)
        final_b = MathTex(r'b', font_size=font_size).next_to(final_a, DOWN * 0.7)
        align_left(final_a, final_b)
        final_ab = Group(final_a, final_b).next_to(arrow, RIGHT * 0.6)


        # Play algo anim
        self.play(*(data_anims + [Write(final_a), Write(final_b)]), run_time=1.5) # 34.5s
        self.play(e.animate.set_value(7), arrow_tracker.animate.set_value(4.5), run_time=5, rate_func=linear)
        self.wait(0.1)


class Action(ThreeDScene):
    def construct(self):
        config.disable_caching = True
        np.random.seed(42)

        x_range = [-6, 6]
        y_range = [-0.5, 1.2]
        x_length = 9
        y_length = 5

        def log_fcn(t):
            return sigmoid(1 * t)

        log_func_graph, log_func_dict = get_func_graph(fcn=log_fcn, x_range=x_range, y_range=y_range, x_length=x_length,
                                                       y_length=y_length, scale=1, font_size=40, include_equation=False,
                                                       curve_length_x_axis_ratio=1)
        log_func_dict["fcn"] = give_rainbow_colors(log_func_dict["fcn"])
        ax = log_func_dict["ax"]
        self.play(Create(log_func_graph))

        self.wait(0.1)

        N = 15
        data_font_size = 25
        self.opacity = 1

        x_data = (np.random.rand(N) - 0.5) * (x_range[1])
        x_data = np.array([math.floor(x * 10) / 10 for x in list(x_data)])
        y_data = np.array([1.1 * i + 1 for i in range(len(x_data))])

        e = ValueTracker(0)

        def get_input(i):
            return MathTex(str(x_data[i]), font_size=data_font_size)#.move_to(pos)

        x_start_shift = 0.5

        def mover(mobj, dt):
            i = mobj.idx
            x = x_data[i]
            min_y = log_fcn(x)
            y = max(y_data[i] - e.get_value(), min_y)
            if y < min_y + 0.000001:
                x_shift = min_y - (y_data[i] - e.get_value())
                x = x + 2 * x_shift
            pos = ax.c2p(x, y)
            if mobj.get_right()[0] > ax.c2p(x_range[1] + x_start_shift)[0] - 0.5:
                mobj.set_opacity(mobj.opacity)
                mobj.set_stroke_opacity(mobj.opacity)
                mobj.set_fill_opacity(mobj.opacity)
                mobj.opacity = max(0, mobj.opacity - 2 * dt)
                if hasattr(mobj, 'tr_path') and mobj.tr_path is not None:
                    if mobj.get_center()[0] > ax.c2p(x_range[1] + x_start_shift + 0.5)[0]:
                        mobj.tr_path.set_opacity(0)
                        mobj.tr_path.set_fill_opacity(0)
                        mobj.tr_path.set_stroke_opacity(0)
            mobj.move_to(pos)

        def get_inputs():
            v = VGroup()
            for i in range(N):
                inp = get_input(i)
                inp.idx = i
                inp.opacity = 1
                inp.add_updater(mover)
                inp.update()
                tr_path = TracedPath(inp.get_center, dissipating_time=0.5, stroke_opacity=[0, 1])
                inp.tr_path = tr_path
                self.add(tr_path)
                v.add(inp)
            return v

        inputs = get_inputs()
        self.add(inputs)

        rect_width = 1
        green_rect_vertices = [[x_range[1] + x_start_shift, 0.5],
                               [x_range[1] + x_start_shift, 1],
                               [x_range[1] + rect_width + x_start_shift, 1],
                               [x_range[1] + rect_width + x_start_shift, 0.5]]
        green_rect = Polygon(*[list(ax.c2p(*v)) for v in green_rect_vertices], fill_color=GREEN, fill_opacity=1,
                             stroke_color=GREEN)
        red_rect = green_rect.copy().shift(ax.c2p(0, 0) - ax.c2p(0, 0.5)).set_color(RED)
        font_size = 45
        strok_width = 5
        label_pass = MathTex(r"\checkmark", color=WHITE, font_size=font_size, stroke_width=strok_width).\
            move_to(green_rect.get_center())
        label_fail = MathTex(r"\times", color=WHITE, font_size=font_size, stroke_width=strok_width).\
            move_to(red_rect.get_center())
        self.play(Create(green_rect), Write(label_pass), Create(red_rect), Write(label_fail), run_time=1)  #1

        self.play(e.animate.set_value(6.6), run_time=7, rate_func=linear)
        self.play(*[FadeOut(obj) for obj in self.mobjects], run_time=1)

        self.wait(0.01)


class LogRes(ThreeDScene):

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_sigmoid_curve(self, ax, x_range, horizontal_squish=1):
        return ParametricFunction(lambda t: ax.c2p(t, self.sigmoid(t * horizontal_squish)), t_range=x_range)

    def get_logistic_func(self, x_range, y_range, x_length, y_length, scale, font_size, formula_shift_up,
                          formula_shift_left):
        arrow_tip_scale = 0.15
        ax = get_axes(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length, scale=scale,
                      arrow_tip_scale=arrow_tip_scale)
        logistic_fcn = self.get_sigmoid_curve(ax=ax, x_range=x_range)
        equation = MathTex(r'f(x) = \frac{1}{1 + e^{-x}}', font_size=font_size). \
            move_to(ax.get_center() + LEFT * formula_shift_left + UP * formula_shift_up).set_color(YELLOW)

        return VGroup(ax, logistic_fcn, equation), {"ax": ax, "logistic_fcn": logistic_fcn, "equation": equation}

    def get_linear_func(self, a, b, x_range, y_range, x_length, y_length, scale):
        arrow_tip_scale = 0.26
        ax = get_axes(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length, scale=scale,
                      arrow_tip_scale=arrow_tip_scale)
        linear_fcn = ParametricFunction(lambda t: ax.c2p(t, a * t + b), t_range=x_range)
        return VGroup(ax, linear_fcn), {"ax": ax, "fcn": linear_fcn}

    def construct(self):
        top_elements_height = 2

        # Sigmoid
        x_range = [-4.5, 4.5]
        y_range = [-0.5, 1.5]
        x_length = 5
        y_length = 3

        logistic_graph_colorful, logistic_graph_colorful_dict = self.get_logistic_func(x_range=x_range, y_range=y_range,
                                                                                       x_length=x_length,
                                                                                       y_length=y_length,
                                                                                       scale=2, font_size=55,
                                                                                       formula_shift_up=1,
                                                                                       formula_shift_left=2.5)
        give_rainbow_colors(logistic_graph_colorful_dict["logistic_fcn"])
        logistic_graph, logistic_graph_dict = self.get_logistic_func(x_range=x_range, y_range=y_range,
                                                                                 x_length=x_length, y_length=y_length,
                                                                                 scale=0.4, font_size=32,
                                                                                 formula_shift_up=0.5,
                                                                                 formula_shift_left=1.5)
        logistic_graph.shift(RIGHT * 4 + UP * (top_elements_height + 0.35))
        give_rainbow_colors(logistic_graph_dict["logistic_fcn"])

        data_x = np.array([-4, -2.3, -1.5, 1.1, 2, 3, 3.7])
        data_y = np.array([0, 0, 0, 1, 1, 1, 1])
        res = stats.linregress(data_x, data_y)
        linear_fcn, linear_fcn_dict = self.get_linear_func(res.slope, res.intercept, x_range, y_range=[-0.9, 1.2],
                                                           x_length=x_length,
                                                           y_length=2.5, scale=1.4)
        linear_fcn.shift(LEFT + DOWN * 0.75)
        give_rainbow_colors(linear_fcn_dict["fcn"])
        points = [Circle(radius=0.05, fill_color=RED, color=RED, fill_opacity=1, stroke_width=2)
                  .move_to(linear_fcn_dict["ax"].c2p(x_, y_)) for x_, y_ in zip(data_x, data_y)]
        point_creation_time = 0.7
        time_to_create_all_points = 3
        start_times = np.random.random(len(points)) * (time_to_create_all_points - point_creation_time)
        point_anims = [Succession(Wait(run_time=st), FadeIn(p, run_time=point_creation_time),
                                  Wait(run_time=time_to_create_all_points - point_creation_time - st))
                       for p, st in zip(points, start_times)]
        logistic_curve_to_transform_into = self.get_sigmoid_curve(linear_fcn_dict["ax"], x_range, horizontal_squish=2)
        give_rainbow_colors(logistic_curve_to_transform_into)

        ################################################################################################
        self.play(*point_anims)

        ax = linear_fcn_dict["ax"]
        font_size = 40
        linear_formula = MathTex(r'f(x) = ', r'ax + b', font_size=font_size). \
            move_to(ax.get_center() + LEFT * 3.1 + UP * 2).set_color(YELLOW)

        self.play(Create(linear_fcn), run_time=2)
        self.play(Write(linear_formula), run_time=1.5)  #6.5

        self.play(Create(logistic_graph_dict["ax"]), run_time=1)  #7.5
        self.play(Write(logistic_graph_dict["logistic_fcn"]), Write(logistic_graph_dict["equation"]), run_time=1.5)  #9

        logistic_ghost = get_ghost(logistic_graph_dict["logistic_fcn"])
        self.play(logistic_ghost.animate.move_to(logistic_curve_to_transform_into.get_center()), run_time=1)
        self.remove(logistic_ghost)
        self.play(Transform(linear_fcn_dict["fcn"], logistic_curve_to_transform_into), run_time=1.5)

        log_res_formula = MathTex(r'f(x) = ', r'\frac{1}{1 + e^{-(ax+b)}}', font_size=font_size). \
            move_to(ax.get_center() + LEFT * 3.6 + UP * 2).set_color(YELLOW)
        align_left(log_res_formula, linear_formula)

        self.play(Transform(linear_formula[1], log_res_formula[1]), run_time=1.5)
        self.wait(2)
        self.play(*[FadeOut(obj) for obj in self.mobjects], run_time=0.5)


class Students(ThreeDScene):

    def get_student_circle(self, radius=0.2):
        return Circle(radius=radius, color=WHITE, fill_opacity=0, stroke_opacity=1, stroke_width=2.5)

    def get_student_name(self, stud, name):
        return Tex(r'\begin{minipage}{15cm}' + name + r'\end{minipage}', font_size=22).next_to(stud, UP * 0.5). \
            set_z_index(1)

    def construct(self):
        v_shift = 0.4
        study_hours = [1.3, 2.5, 3.7, 5, 6, 7, 8.5]
        results = [0, 0, 0, 1, 1, 1, 1]

        names = ["Bob", "Alice", "Nick", "Marta", "Rob", "Carol", "Max", "Lea"]

        x_data = study_hours
        y_data = results
        res = stats.linregress(np.array(x_data), np.array(y_data))

        ################################ Main graph
        x_range = [0, 9.8]
        y_range = [0, 1.75]
        x_length = 5.5
        y_length = 2
        hours_studied_graph, hours_studied_vgroup, hours_studied_graph_dict = get_coordinate_line(scene_obj=self,
                                                                                                  x_range=x_range,
                                                                                                  y_range=y_range,
                                                                                                  x_length=x_length,
                                                                                                  y_length=y_length,
                                                                                                  scale=1.2,
                                                                                                  arrow_tip_scale=0.2)
        ax = hours_studied_graph_dict["original_coords"]
        x_label = Tex(r'\begin{minipage}{15cm}Hours studied\end{minipage}', font_size=35). \
            next_to(hours_studied_graph, DOWN).shift(RIGHT * 3.3)
        y_axis = hours_studied_graph_dict["y_coord_line"]
        y_label = Tex(r'y', font_size=35).next_to(y_axis, LEFT * 0.3).shift(UP * 0.2)
        all_parts_of_graph = VGroup(hours_studied_vgroup, y_axis, x_label)
        student_circles = [self.get_student_circle().move_to(hours_studied_graph_dict["original_coords"].
                                                             c2p(hour, 0, 0)).shift(DOWN) for hour in study_hours]

        all_parts_of_graph.shift(DOWN)
        student_circles = VGroup(*student_circles)
        all_parts_of_graph.add(student_circles)
        study_hours_anims = [Create(stud) for stud in student_circles]

        result_labels = [get_student_label(stud, result) for stud, result in zip(student_circles, results)]
        for lab in result_labels:
            all_parts_of_graph.add(lab)

        fitted_straight_line = ParametricFunction(lambda t: hours_studied_graph_dict["original_coords"].
                                                  c2p(t, res.slope * t + res.intercept), t_range=x_range)
        fitted_straight_line = give_rainbow_colors(fitted_straight_line)
        all_parts_of_graph.add(fitted_straight_line)
        all_parts_of_graph.add(y_label)
        all_parts_of_graph.scale(1.05).shift(DOWN * 0.2).shift(RIGHT)

        student_names = [self.get_student_name(stud_circle, name) for stud_circle, name in zip(student_circles, names)]
        student_names = VGroup(*student_names)
        all_parts_of_graph.add(student_circles)
        all_parts_of_graph.add(student_names)
        name_anims = [Write(name) for name in student_names]

        self.play(Create(hours_studied_graph), Write(x_label), Create(y_axis), Write(y_label), run_time=1.5)
        self.play(*(study_hours_anims + name_anims), run_time=1.5)
        self.wait(3)  #6

        ################################ Small linear function
        graph_x_range = [-4, 4]
        graph_y_range = [-3, 3]
        graph_x_length = 2.8
        graph_y_length = 1.9
        graph_scale = 1

        def linear_fcn(xx):
            return res.slope * xx + res.intercept

        font_size = 20
        linear_fcn_vgroup, linear_fcn_dict = get_func_graph(fcn=linear_fcn, x_range=graph_x_range,
                                                     y_range=graph_y_range,
                                                     x_length=graph_x_length,
                                                     y_length=graph_y_length, scale=graph_scale,
                                                     font_size=font_size,
                                                     include_equation=False)
        linear_fcn_vgroup.move_to(np.array([-4.5, 2.5, 0])).shift(DOWN * v_shift)
        equation = MathTex(r'y = ', r'ax + b', font_size=24, color=YELLOW).next_to(linear_fcn_dict["ax"], RIGHT * 0.1). \
            shift(UP * 1 + LEFT * 0.5)
        axis_labels = linear_fcn_dict["ax"].get_axis_labels(x_label='x', y_label='y')
        label_scale = 0.75
        axis_labels[0].scale(label_scale)
        axis_labels[1].scale(label_scale).shift(LEFT * 0.5)
        linear_fcn_vgroup.add(equation)
        linear_fcn_vgroup.add(axis_labels)
        linear_fcn_dict["fcn"] = give_rainbow_colors(linear_fcn_dict["fcn"])
        straight_line_to_transform = linear_fcn_dict["fcn"].copy()
        straight_line_to_transform.set_opacity(0.5)
        straight_line_to_transform = give_rainbow_colors(straight_line_to_transform)
        linear_part_small = get_ghost(equation[1], opacity=1)

        ################################ Params
        param_def = MathTex(r'\text{Parameters}&\hphantom{:}\\',
                            r'\text{Slope}', r'&:', r'a', r' = ' + str("{:.1f}".format(res.slope)) + r'\\',
                            r'\text{Intercept}', r'&:', r'b', r' = ' + str("{:.1f}".format(res.intercept)),
                            font_size=28, color=YELLOW).shift(LEFT * 5.1)
        param_def[0].set_color(WHITE)
        param_def[1].set_color(WHITE)
        param_def[2].set_color(WHITE)
        param_def[5].set_color(WHITE)
        param_def[6].set_color(WHITE)
        self.play(Write(param_def[0]), Write(param_def[1]), Write(param_def[2]), Write(param_def[3]),
                  Write(param_def[5]), Write(param_def[6]), Write(param_def[7]), Create(linear_fcn_vgroup),
                  Write(param_def[4]), Write(param_def[8]), run_time=1.3)  # 7.3

        self.play(Transform(straight_line_to_transform, fitted_straight_line), run_time=1.4)

        ############################## Move points to the straight line.
        results_on_line_y = [res.slope * x + res.intercept for x in x_data]
        stud_positions_on_line = [hours_studied_graph_dict["original_coords"].
                                  c2p(x, y) for x, y in zip(study_hours, results_on_line_y)]
        stud_shifts = [new_pos - lab.get_center() for new_pos, lab in zip(stud_positions_on_line, result_labels)]
        stud_linear_shift_anims = [circ.animate.shift(shift) for circ, shift in zip(student_circles, stud_shifts)]
        stud_name_shift_anims = [res.animate.shift(shift) for res, shift in zip(student_names, stud_shifts)]
        self.play(*(stud_linear_shift_anims + stud_name_shift_anims), run_time=1.3)  #10

        def log_func(t):
            return 1 / (1 + np.exp(-0.5 * t))

        ################################ Logistic graph
        logistic_fcn, logistic_fcn_dict = get_func_graph(fcn=log_func, x_range=[-8, 8],
                                                         y_range=[-0.5, 1.2],
                                                         x_length=graph_x_length,
                                                         y_length=graph_y_length, scale=graph_scale,
                                                         font_size=font_size,
                                                         include_equation=False)
        logistic_fcn.move_to(np.array([4.5, 2.5, 0])).shift(DOWN * v_shift)
        equation = MathTex(r'z = ', r'\frac{1}{1 + e', r'^{-y}}', font_size=24, color=YELLOW).next_to(logistic_fcn_dict["ax"],
                                                                                            RIGHT * 0.1). \
            shift(UP * 0.9 + LEFT * 0.5)
        axis_labels = logistic_fcn_dict["ax"].get_axis_labels(x_label='y', y_label='z')
        logistic_fcn_dict["fcn"] = give_rainbow_colors(logistic_fcn_dict["fcn"])
        label_scale = 0.75
        axis_labels[0].scale(label_scale)
        axis_labels[1].scale(label_scale).shift(LEFT * 0.5)
        logistic_fcn.add(equation)
        logistic_fcn.add(axis_labels)

        self.play(Create(logistic_fcn), run_time=1)  #11

        ################################ Merge functions
        main_function = MathTex(r'\frac{1}{1 + e', r'^{-(', r'ax + b', r')}}',
                                font_size=40, color=YELLOW).shift(UP * 3)
        main_function.shift(DOWN * v_shift * 2)
        sigmoid_part = main_function[0]
        linear_part = main_function[2]
        sigmoid_part_small = get_ghost(equation[1], opacity=1)

        def logistic_func_big(t):
            val = 1 / (1 + np.exp(-(1.2 * t - x_range[1] / 2)))
            print("val: " + str(val))
            return val

        logistic_ghost = get_ghost(logistic_fcn, opacity=0.6)

        def scale_down(mobj, dt):
            mobj.scale(1 - dt * 0.4)

        logistic_ghost.add_updater(scale_down)
        self.play(logistic_ghost.animate.move_to(straight_line_to_transform.get_center()), run_time=1)  # 12
        self.remove(logistic_ghost)

        logistic_curve_big = ParametricFunction(lambda t: hours_studied_graph_dict["original_coords"].
                                                c2p(t, logistic_func_big(t)), t_range=x_range)
        logistic_curve_big = give_rainbow_colors(logistic_curve_big)

        results_on_line_y = [logistic_func_big(x) for x in study_hours]
        stud_positions_on_line = [hours_studied_graph_dict["original_coords"].
                                  c2p(x, y) for x, y in zip(study_hours, results_on_line_y)]
        stud_shifts = [new_pos - lab.get_center() for new_pos, lab in zip(stud_positions_on_line, student_circles)]
        stud_linear_shift_anims = [circ.animate.shift(shift) for circ, shift in zip(student_circles, stud_shifts)]
        stud_name_shift_anims = [res.animate.shift(shift) for res, shift in zip(student_names, stud_shifts)]

        anims = stud_linear_shift_anims + stud_name_shift_anims + [Transform(sigmoid_part_small, sigmoid_part),
                                                                   Transform(linear_part_small, linear_part),
                  Transform(straight_line_to_transform, logistic_curve_big)]

        self.play(*anims, run_time=1)  # 13
        self.play(Write(main_function[1]), Write(main_function[3]), run_time=0.6)  # 15.1

        line_start = ax.c2p(0, 0.5)
        line_end = ax.c2p(x_range[1], 0.5)

        line_at_level_05 = DashedLine(start=line_start, end=line_end, color=YELLOW, stroke_width=stroke_width_line05)
        y_equals_05_label = Tex(r'y = 0.5', font_size=35, color=YELLOW).next_to(line_at_level_05, RIGHT)

        # self.wait(0.5)
        self.play(Write(line_at_level_05), Write(y_equals_05_label), run_time=1.5)

        for lab, stud in zip(result_labels, student_circles):
            lab.move_to(stud.get_center()).scale(0.9)
        result_labels_anims_pass = [Write(lab) for lab in result_labels[3:]]
        result_labels_anims_fail = [Write(lab) for lab in result_labels[:3]]

        self.play(*result_labels_anims_pass, run_time=1)
        self.wait()
        self.play(*result_labels_anims_fail, run_time=1)
        self.wait()

        # self.remove(logistic_ghost)
        # self.remove(linear_part)
        # self.remove(linear_part_small)
        # self.remove(sigmoid_part)
        # self.remove(sigmoid_part_small)

        line_start = ax.c2p(0, 1)
        line_end = ax.c2p(x_range[1], 1)
        line_at_level_1 = DashedLine(start=line_start, end=line_end, color=YELLOW, stroke_width=stroke_width_line05)
        y_equals_1_label = Tex(r'y = 1', font_size=35, color=YELLOW).next_to(line_at_level_1, RIGHT)
        anims = [Unwrite(stud) for stud in student_names] + [Uncreate(stud) for stud in student_circles] + \
        [FadeOut(linear_fcn_vgroup), FadeOut(logistic_fcn), FadeOut(main_function), Write(line_at_level_1),
         Write(y_equals_1_label)]

        # self.remove(logistic_ghost)
        # self.remove(linear_part)
        # self.remove(linear_part_small)
        # self.remove(sigmoid_part)
        # self.remove(sigmoid_part_small)
        anims += [FadeOut(m) for m in [logistic_ghost, linear_part, linear_part_small, sigmoid_part, sigmoid_part_small]]

        self.wait(3)
        self.play(*anims, run_time=1)  #2

        ################################ Cost function
        font_size = 35
        h = 2.4
        cost_function = MathTex(r'L(a, b)', r'&=',
                                r'-', r' \frac{1}{m} \sum_{i=1}^{m}\left[y_i \log(h(x_i)) + (1 - y_i) \log(1 - h(x_i))\right]\;,\\',
                                r'h(x_i)', r'&=', r'\frac{1}{1 + e^{-(a x_i + b)}}',
                                font_size=font_size, color=BLUE).shift(UP * h)
        self.play(Write(cost_function), run_time=2)  #6
        self.wait()  #7

        # Money
        money_img = ImageMobject("data/money.png").scale(0.35).shift(UP * 1.2 + RIGHT * 2.5)
        self.wait(2.5)
        self.play(FadeIn(money_img), run_time=1.5)
        self.wait(1)  #12

        # Highlight a and b
        a = cost_function[0][2]
        b = cost_function[0][4]
        a2 = cost_function[6][7]
        b2 = cost_function[6][11]
        a_highlight = MathTex(r'a', color=YELLOW, font_size=font_size).move_to(a)
        b_highlight = MathTex(r'b', color=YELLOW, font_size=font_size).move_to(b)
        a2_highlight = MathTex(r'a', color=YELLOW, font_size=font_size).move_to(a2)
        b2_highlight = MathTex(r'b', color=YELLOW, font_size=font_size).move_to(b2)
        self.wait(3)
        self.play(FadeOut(a), Write(a_highlight), FadeOut(b), Write(b_highlight), FadeOut(a2), Write(a2_highlight),
                  FadeOut(b2), Write(b2_highlight), run_time=1.5)
        self.wait(0.5)  #16

        ####################################  add min_{a, b} in front of L(a, b) and the formula
        cost_function_2 = MathTex(r'\min_{a, b} L(a, b)', r'&=',
                                r'\min_{a, b}', r'-', r' \frac{1}{m} \sum_{i=1}^{m}\left[y_i \log(h(x_i)) + (1 - y_i) \log(1 - h(x_i))\right]\;,\\',
                                r'h(x_i)', r'&=', r'\frac{1}{1 + e^{-(a x_i + b)}}',
                                font_size=font_size, color=BLUE).shift(UP * h)
        shift_ = cost_function[2].get_center() - cost_function_2[3].get_center()
        cost_function_2.shift(shift_)
        shift_ = cost_function_2[0][-1].get_center() - cost_function[0][-1].get_center()
        vg = VGroup(cost_function[0][:2], cost_function[0][3], cost_function[0][5:])
        self.wait(8)
        self.play(vg.animate.shift(shift_), cost_function[1].animate.shift(shift_),
                  a_highlight.animate.shift(shift_), b_highlight.animate.shift(shift_), run_time=1)
        self.play(Write(cost_function_2[0][:6]), Write(cost_function_2[2]), run_time=1)  #27

        # Highlight a and b
        a = cost_function_2[0][3]
        b = cost_function_2[0][5]
        a2 = cost_function_2[2][3]
        b2 = cost_function_2[2][5]
        a_highlight = MathTex(r'a', color=YELLOW, font_size=font_size).move_to(a)
        b_highlight = MathTex(r'b', color=YELLOW, font_size=font_size).move_to(b)
        a2_highlight = MathTex(r'a', color=YELLOW, font_size=font_size).move_to(a2)
        b2_highlight = MathTex(r'b', color=YELLOW, font_size=font_size).move_to(b2)
        self.play(FadeOut(a), Write(a_highlight), FadeOut(b), Write(b_highlight), FadeOut(a2), Write(a2_highlight),
                  FadeOut(b2), Write(b2_highlight), run_time=1)  # 28

        ########### Write error bars

        def horizontal_bar(point, color):
            buff = 0.07
            start_ = point + LEFT * buff
            end_ = point + RIGHT * buff
            return Line(start=start_, end=end_, color=color, buff=0.0)

        def get_vert_bar(x, pos, is_above):
            v_buff = 0.008
            start_ = hours_studied_graph_dict["original_coords"].c2p(x, logistic_func_big(x))
            # start_ = pos
            yval = 1 if is_above else 0
            end_ = hours_studied_graph_dict["original_coords"].c2p(x, yval)
            end_[0] = start_[0]
            if is_above:
                start_ = start_ + v_buff * DOWN
                end_ = end_ + v_buff * UP
            else:
                start_ = start_ + v_buff * UP
                end_ = end_ + v_buff * DOWN
            color = WHITE
            line = Line(start=start_, end=end_, color=color, buff=0.0)
            hor_bar1 = horizontal_bar(start_, color)
            hor_bar2 = horizontal_bar(end_, color)
            return VGroup(line, hor_bar1, hor_bar2)

        above05 = [False, False, False, True, True, True, True]
        bars = [get_vert_bar(x, label.get_center(), above) for x, label, above in zip(study_hours, student_circles, above05)]
        anims = [Transform(label, bar) for bar, label in zip(bars, result_labels)]
        self.wait(7)
        self.play(*anims, run_time=1.5)
        self.wait(1)  #37.5

        ####################### Write errors
        errors = [abs(logistic_func_big(x) - float(y)) for x, y in zip(study_hours, above05)]

        def format_error(val):
            return str(val)[:4]

        def get_error_position(arr, is_above):
            v_shift = 0.3
            if is_above:
                return arr.get_top() + v_shift * UP
            else:
                return arr.get_bottom() + v_shift * DOWN

        formula_color = BLUE
        error_labels = [MathTex(str(format_error(error)), font_size=formula_font).set_color(formula_color).
                        move_to(get_error_position(arr, above)) for error, arr, above in
                        zip(errors, bars, above05)]

        highlight_time = 0.3
        arrow_time = 0.3
        error_label_time = 0.3
        error_strings = [str(format_error(error)) for error in errors]
        formula_strings = []
        for idx, err in enumerate(error_strings):
            formula_strings.append(err)
            formula_strings.append("^2")
            if idx < len(error_strings) - 1:
                formula_strings.append("+")

        formula = MathTex(r'Loss = ', *formula_strings, font_size=formula_font).next_to(ax, DOWN * 4.5).set_color(
            YELLOW).shift(UP * 0.5).shift(LEFT * 4.5)
        formula_left = formula.get_left()
        first_error_coords = formula[1].get_center()
        errors_inside_formula = [formula[1 + 3 * i] for i in range(len(errors))]
        error_positions_inside_formula = [err.get_center() for err in errors_inside_formula]
        for i, err in enumerate(errors_inside_formula):
            err.move_to(error_labels[i])
        # errors_inside_formula_without_left_most = [err for i, err in enumerate(errors_inside_formula) if i != left_most]
        error_move_starting_times = [i * 0.3 for i in range(len(errors_inside_formula))]
        error_moving_time = 0.8
        plus_writing_time = 0.3
        power_write_time = 0.6
        formula_shift_time = 0.7
        pluses = [formula[3 + 3 * i] for i in range(len(study_hours) - 1)]
        pluses.append(MathTex("1").move_to(np.array([100, 100, 0])))
        formula_new = MathTex(r'Loss = ', r'\frac{1}{' + str(len(study_hours)) + r'}\Big(', *formula_strings, r'\Big)',
                              font_size=formula_font).set_color(YELLOW)
        formula_new.shift(formula_left - formula_new.get_left())
        formula_shift = formula_new[2].get_center() - first_error_coords
        formula_parts_to_shift = errors_inside_formula + pluses
        powers = [formula_new[3 + 3 * i] for i in range(len(study_hours))]
        for p in powers:
            p.set_color(BLUE)
        n_people = MathTex(str(len(study_hours)) + r'\text{people}', font_size=formula_font)
        n_people.shift(formula[0].get_left() - n_people.get_left() + DOWN * 0.7)

        self.wait()
        self.play(*[Succession(Wait(run_time=t), error.animate(run_time=error_moving_time).move_to(pos),
                               Write(plus, run_time=plus_writing_time)) for error, t, pos, plus in
                    zip(errors_inside_formula, error_move_starting_times, error_positions_inside_formula, pluses)])

        self.play(*[p.animate(run_time=formula_shift_time).shift(formula_shift) for p in formula_parts_to_shift])
        self.play(Write(formula_new[1]), Write(formula_new[-1]), run_time=1)
        self.wait(0.1)
        self.play(*[Write(p, run_time=power_write_time) for p in powers])

        anims = [FadeOut(mobj) for mobj in self.mobjects]
        self.wait(2)
        self.play(*anims, run_time=0.5)

        self.wait(0.1)


class GradDescent(ThreeDScene):

    def __init__(self):
        super().__init__()
        self.f = None
        self.f_da = None
        self.f_db = None
        self.x_data = [1.3, 2.5, 3.7, 5, 6, 7, 8.5]
        self.y_data = [0, 0, 0, 1, 1, 1, 1]

    def log_res_probability(self, a, b, x):
        return 1 / (1 + exp(-(a * x + b)))
    def cost_single_data_point(self, a, b, x, y):
        assert (y == 1 or y == 0)
        return (y - self.log_res_probability(a, b, x))**2

    def get_symbollic_atoms(self):
        a = Symbol('a')
        b = Symbol('b')
        m = len(self.x_data)
        res = 0
        for x, y in zip(self.x_data, self.y_data):
            res += self.cost_single_data_point(a, b, x, y)
        y = res / m
        return a, b, y

    def get_func_and_derivatives(self):
        a, b, y = self.get_symbollic_atoms()
        diff_a = y.diff(a)
        diff_b = y.diff(b)
        return lambdify([a, b], y, 'numpy'), lambdify([a, b], diff_a, 'numpy'), lambdify([a, b], diff_b, 'numpy')

    def construct(self):
        a_range = [-50, 50]
        b_range = [-50, 50]
        L_range = [0, 1]
        a_length = 6
        b_length = 6
        L_length = 10
        scale=1

        ################## Axes
        ax = get_3d_axes(x_range=a_range, y_range=L_range, z_range=b_range,
                         x_length=a_length, y_length=L_length, z_length=b_length, scale=0.6, arrow_tip_scale=None).\
            rotate(axis=UP, angle=30 * DEGREES).rotate(axis=RIGHT, angle=20 * DEGREES).\
            shift(DOWN * 1.5)
        x_label = ax.get_x_axis_label('a')
        y_label = ax.get_y_axis_label('L(a, b)')
        z_label = ax.get_z_axis_label('b')
        self.play(Create(ax), Write(x_label), Write(y_label), Write(z_label))

        ################## Cost function graph

        self.f, self.f_da, self.f_db = self.get_func_and_derivatives()
        graph = Surface(lambda a, b: ax.c2p(a, self.f(a, b), b), u_range=a_range, v_range=b_range)
        self.play(Create(graph))

        def rotator(mob, dt):
            mob.rotate(axis=UP, angle=dt)

        f_graph = VGroup(ax, graph, x_label, y_label, z_label)
        self.add(f_graph)
        f_graph.add_updater(rotator)
        # graph.add_updater(rotator)

        self.wait(4)


formula_font = 28
big_formula_font = 60

def get_params():
    min_x = -7
    max_x = 7
    n_points_at_once = 60
    n_updates = 20
    n_points_to_update_per_turn = 30
    init_a = 0.4
    init_b = 0
    disturb_coeff = 1
    point_r = 0.04
    point_color = RED
    point_removal_mode = "oldest"
    return min_x, max_x, n_points_at_once, n_updates, n_points_to_update_per_turn, init_a, init_b, disturb_coeff, \
           point_r, point_color, point_removal_mode


def get_grid(rows, cols, cell_width, cell_height, shift=ORIGIN):
    v = VGroup()

    # Create the grid lines
    for i in range(rows + 1):
        start = (i - rows / 2) * cell_height * UP + (cols / 2) * cell_width * LEFT
        end = (i - rows / 2) * cell_height * UP + (cols / 2) * cell_width * RIGHT
        line = Line(start, end)
        v.add(line)

    for i in range(cols + 1):
        start = (i - cols / 2) * cell_width * RIGHT + (rows / 2) * cell_height * UP
        end = (i - cols / 2) * cell_width * RIGHT + (rows / 2) * cell_height * DOWN
        line = Line(start, end)
        v.add(line)

    v.shift(shift)

    # Calculate cell centers
    centers = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append((j - (cols - 1) / 2) * cell_width * RIGHT
                       + (i - (rows - 1) / 2) * cell_height * UP
                       + shift)
        centers.append(row)
    return {"grid": v, "cell_center_positions": centers}


class MLE(ThreeDScene):
    def construct(self):
        config.disable_caching = True

        ######################### Text at the top
        mle_text = MathTex(r'\text{Maximum likelihood estimator}', font_size=big_formula_font * 0.85).\
            set_color(YELLOW).shift(UP * 3.2)
        self.play(Write(mle_text), run_time=1.3)

        ######################### Graph

        x_range = [0, 9.8]
        y_range = [0, 1.3]
        x_length = 5.5
        y_length = 2.1
        hours_studied_graph, hours_studied_vgroup, hours_studied_graph_dict = get_coordinate_line(scene_obj=self,
                                                                                                  x_range=x_range,
                                                                                                  y_range=y_range,
                                                                                                  x_length=x_length,
                                                                                                  y_length=y_length,
                                                                                                  scale=1.2,
                                                                                                  arrow_tip_scale=0.2)
        ax = hours_studied_graph_dict["original_coords"]
        x_label = Tex(r'\begin{minipage}{15cm}Hours studied\end{minipage}', font_size=35). \
            next_to(hours_studied_graph, DOWN).shift(RIGHT * 3.3)
        y_axis = hours_studied_graph_dict["y_coord_line"]
        y_label = Tex(r'y', font_size=35).next_to(y_axis, LEFT * 0.3).shift(UP * 0.2)
        all_parts_of_graph = VGroup(hours_studied_vgroup, y_axis, x_label, y_label)

        label_font_size = 35
        font_size_scale = 1
        label_color = PURPLE
        line_start = hours_studied_graph.get_left() + hours_studied_graph_dict["unit_vec_y"]
        line_end = hours_studied_graph.get_right() + hours_studied_graph_dict["unit_vec_y"]
        line_at_level_1 = DashedLine(start=line_start, end=line_end, color=YELLOW, stroke_width=stroke_width_line05)
        all_parts_of_graph.add(line_at_level_1)
        y_equals_1_label = Tex(r'\boldmath{$y = 1$}', font_size=label_font_size * font_size_scale, color=label_color).next_to(line_at_level_1, RIGHT)
        all_parts_of_graph.add(y_equals_1_label)

        line_start = ax.c2p(0, 0.5, 0)
        line_end = ax.c2p(x_range[1], 0.5, 0)
        line_at_level_05 = DashedLine(start=line_start, end=line_end, color=YELLOW, stroke_width=stroke_width_line05)
        all_parts_of_graph.add(line_at_level_05)
        y_equals_05_label = Tex(r"\boldmath{$y = 0.5$}", font_size=label_font_size * font_size_scale, color=label_color).next_to(line_at_level_05, RIGHT)
        all_parts_of_graph.add(y_equals_05_label)
        # self.wait(0.5)
        self.play(Create(hours_studied_graph), Write(x_label), Create(y_axis), Write(y_label),
                  Create(line_at_level_1), Write(y_equals_1_label),
                  Create(line_at_level_05), Write(y_equals_05_label), run_time=1.3)  #4

        ################################# Logistic curve
        def logistic_func_big(t):
            return 1 / (1 + np.exp(-(1.3 * t - x_range[1] / 2)))

        logistic_curve_big = ParametricFunction(lambda t: hours_studied_graph_dict["original_coords"].
                                                c2p(t, logistic_func_big(t)), t_range=x_range)
        logistic_curve_big = give_rainbow_colors(logistic_curve_big)
        all_parts_of_graph.add(logistic_curve_big)
        log_formula = MathTex(r'\frac{1}{1 + e^{-(a x + b)}}', font_size=big_formula_font * 0.65).set_color(
            YELLOW).next_to(logistic_curve_big, UP * 2)
        all_parts_of_graph.add(log_formula)
        self.play(Write(logistic_curve_big), Write(log_formula), run_time=1.3)  #4

        ################################# Sliding point
        x_val_tracker = ValueTracker(2.5)
        point = always_redraw(lambda: Circle(radius=0.05, fill_color=YELLOW).set_color(YELLOW).
                              move_to(hours_studied_graph_dict["original_coords"].c2p(x_val_tracker.get_value(),
                                                                                      logistic_func_big(
                                                                                          x_val_tracker.get_value()),
                                                                                      0)))
        bar = always_redraw(lambda: Line(
            start=hours_studied_graph_dict["original_coords"].c2p(x_val_tracker.get_value(),
                                                                  logistic_func_big(
                                                                      x_val_tracker.get_value()),
                                                                  0),
            end=hours_studied_graph_dict["original_coords"].c2p(x_val_tracker.get_value(), 0, 0)
        )
                            )

        distance_h_shift = 1.3

        def get_prob_of_1():
            return str(logistic_func_big(x_val_tracker.get_value()))[:4]

        distance = always_redraw(lambda: MathTex(r'p(1|x) = ' + get_prob_of_1(), font_size=25).
                                 set_color(YELLOW).move_to(
            hours_studied_graph_dict["original_coords"].c2p(x_val_tracker.get_value() + distance_h_shift,
                                                            logistic_func_big(
                                                                x_val_tracker.get_value()) * 0.77, 0)))

        self.wait(0.5)
        self.play(Create(point), Write(bar), run_time=1.5)
        self.play(Write(distance), run_time=1.5)
        self.play(x_val_tracker.animate.set_value(x_range[1] - 2.5), run_time=3.5)  #11

        ################################# Unknown points
        unknown_x = [1.5, 3.5, 5.5, 7.5]
        ax_ = hours_studied_graph_dict["original_coords"]
        unknown_points = [Text("?", weight=BOLD, font_size=label_font_size).set_color(PURPLE).
                          move_to(ax_.c2p(x, 0.5, 0)) for x in unknown_x]
        anims = [Write(p) for p in unknown_points]
        self.wait(2.5)
        self.play(*anims, run_time=1.5)  #15

        ################################# Remove stuff
        self.play(Uncreate(point), Unwrite(bar), Unwrite(distance), Unwrite(log_formula))

        ################################# Params a, b
        ab_formula = MathTex(r'a, b', font_size=big_formula_font * 0.65).set_color(
            YELLOW).next_to(
            logistic_curve_big,
            LEFT * 2 + UP * 1)
        all_parts_of_graph.add(ab_formula)
        self.play(Write(ab_formula))  #17

        ################################# Remove the unknown points
        anims = [Uncreate(p) for p in unknown_points]
        self.wait(0.5)
        self.play(*anims)  #18.5

        ################################# Training data
        train_xy = [(0.3, 0), (1.2, 0), (2.1, 1), (2.9, 0), (4.6, 1), (5.5, 1), (6.1, 0), (6.9, 1), (8.1, 1), (9.1, 1)]
        ax_ = hours_studied_graph_dict["original_coords"]
        coord_dummies = [Circle(radius=0.01).move_to(ax_.c2p(xy[0], xy[1], 0)) for xy in train_xy]
        train_points = [get_student_label(coord, xy[1], numeric=False, font_size=30)
                  for xy, coord in zip(train_xy, coord_dummies)]
        all_parts_of_graph.add(*train_points)

        def get_shift(x):
            x = int(x)
            if x == 0:
                return -1
            elif x == 1:
                return 1
            else:
                raise ValueError("Wrong x")

        x_coord_font_size = 30
        x_coords = [MathTex(str(xy[0]), font_size=x_coord_font_size, color=YELLOW).move_to(p.get_center()).
                    shift(0.47 * UP * get_shift(xy[1]))
                    for xy, p in zip(train_xy, train_points)]
        all_parts_of_graph.add(*x_coords)
        anims = [Write(p) for p in train_points] + [Write(x) for x in x_coords]
        self.wait(1)
        self.play(*anims, run_time=1.5)  #21

        ################################# Draw circles around the ticks
        def draw_circle_around_point(p, radius, stroke_width, color):
            return Circle(radius=radius, stroke_width=stroke_width, color=color).move_to(p)

        radius = 0.25
        stroke_width = 3
        color = WHITE
        circles = [draw_circle_around_point(p=p.get_center(), radius=radius, stroke_width=stroke_width, color=color)
                   for p in train_points]
        ticks = [circle for circle, xy in zip(circles, train_xy) if xy[1] == 1]
        anims = [Write(c) for c in ticks]
        self.wait(7.8)
        self.play(*anims)
        self.wait(0.1)
        anims = [Unwrite(c) for c in ticks]
        self.play(*anims)
        self.wait(0.1)

        ################################# Draw circles around the crosses
        crosses = [circle for circle, xy in zip(circles, train_xy) if xy[1] == 0]
        self.wait(0.5)
        anims = [Write(c) for c in crosses]
        self.play(*anims)
        self.wait(0.1)
        anims = [Unwrite(c) for c in crosses]
        self.play(*anims)  #33.7

        ################################# Draw circles around the x-coords
        x_coord_circles = [draw_circle_around_point(p=x.get_center(), radius=radius, stroke_width=stroke_width, color=color)
                   for x in x_coords]
        self.wait(0.7)
        anims = [Write(c) for c in x_coord_circles]
        self.play(*anims)
        self.wait(0.1)
        anims = [Unwrite(c) for c in x_coord_circles]
        self.play(*anims)
        self.wait(0.7)

        ################################# Move the graph
        graph_scale = 0.7
        def apply_function(m):
            m.scale(graph_scale)
            m.shift(RIGHT * 3 + UP * 1.5)
            return m

        self.play(ApplyFunction(apply_function, all_parts_of_graph), run_time=1.5)
        self.wait(0.1)

        ################################# Draw the x-grid
        rows = len(train_xy)

        # Cell size
        cell_width_x = 0.46
        cell_height = 0.46
        shift_x = 5.7
        label_font_size = 35
        label_vert_offset = 0.25

        x_dict = get_grid(rows=rows, cols=1, cell_width=cell_width_x, cell_height=cell_height, shift=LEFT * shift_x)
        x_label = MathTex(r'x', font_size=label_font_size).shift(LEFT * shift_x +
                                                                 + cell_height * rows / 2 * UP +
                                                                 UP * label_vert_offset)
        self.play(Write(x_dict["grid"]), Write(x_label), run_time=1.5)

        ################################# Move the x-coords
        delta_time = 0.1
        start_times = [i * delta_time for i in range(len(x_coords))]
        point_anims = [Succession(Wait(run_time=st), x.animate.move_to(pos))
                       for x, st, pos in zip(x_coords, start_times, x_dict["cell_center_positions"])]
        self.play(*point_anims, run_time=1.9)  #42

        ################################# Draw the y-grid
        cell_width_y = 0.46
        hor_offset = 0.3
        shift_y = shift_x - (cell_width_x/2 + cell_width_y/2 + hor_offset)

        y_dict = get_grid(rows=rows, cols=1, cell_width=cell_width_y, cell_height=cell_height, shift=LEFT * shift_y)
        y_label = MathTex(r'y', font_size=label_font_size).shift(LEFT * shift_y +
                                                                 + cell_height * rows / 2 * UP +
                                                                 UP * label_vert_offset)

        self.play(Write(y_dict["grid"]), Write(y_label), run_time=1.5)

        ################################# Move the y-coords
        delta_time = 0.1
        start_times = [i * delta_time for i in range(len(x_coords))]
        y_label_ghosts = [y.copy() for y in train_points]
        y_label_anims = [Succession(Wait(run_time=st), y.animate.move_to(pos))
                       for y, st, pos in zip(y_label_ghosts, start_times, y_dict["cell_center_positions"])]
        self.play(*y_label_anims, run_time=1.9)  #45.5

        ################################# Draw the p(y|x)-label
        cell_width_p = 1.2
        shift_p = shift_y - (cell_width_y/2 + cell_width_p/2 + hor_offset)

        p_dict = get_grid(rows=rows, cols=1, cell_width=cell_width_p, cell_height=cell_height, shift=LEFT * shift_p)
        p_label = MathTex(r'p(y \vert x)', font_size=label_font_size).shift(LEFT * shift_p +
                                                                 + cell_height * rows / 2 * UP +
                                                                 UP * label_vert_offset)
        self.wait(2)
        self.play(Write(p_label), run_time=1.5)  #50
        self.wait(1)

        ################################# Draw the p(y|x)-grid
        cell_id = -1
        p_formula_color = WHITE
        pxy = [MathTex(r"p(", r"\cross", r"\vert", r"4.5", r")", font_size=x_coord_font_size, color=p_formula_color).
               scale(graph_scale * 1.2).move_to(pos)
                    for pos in p_dict["cell_center_positions"]]
        self.wait(8.5)
        self.play(Write(p_dict["grid"]), Write(pxy[cell_id][0]), Write(pxy[cell_id][2]),
                  Write(pxy[cell_id][4]), run_time=2)

        ################################# Animate the first student results
        x_ghost = x_coords[cell_id].copy()
        self.play(x_ghost.animate.move_to(pxy[cell_id][3]), run_time=1.5)  #1:02
        y_ghost_of_ghost = y_label_ghosts[cell_id].copy()
        self.wait(6.5)
        self.play(y_ghost_of_ghost.animate.move_to(pxy[cell_id][1]), run_time=1.5)  #1:10

        ################################# Animate the last student results
        cell_id = 0
        self.wait(10.5)
        self.play(Write(pxy[cell_id][0]), Write(pxy[cell_id][2]),
                  Write(pxy[cell_id][4]))
        x_ghost = x_coords[cell_id].copy()
        self.play(x_ghost.animate.move_to(pxy[cell_id][3]), run_time=1.5)  #(1:23)
        y_ghost_of_ghost = y_label_ghosts[cell_id].copy()
        self.wait(1.5)
        self.play(y_ghost_of_ghost.animate.move_to(pxy[cell_id][1]), run_time=1.5)  #(1:26)

        ################################# Animate the remaining student results
        delta_time = 0.1
        start_times = [i * delta_time for i in range(len(x_coords[1:-1]))]
        x_ghosts = [x_coords[cell_id].copy() for cell_id in range(1, len(x_coords) - 1)]
        y_ghosts_of_ghosts = [y_label_ghosts[cell_id].copy() for cell_id in range(1, len(x_coords) - 1)]
        point_anims = [Succession(Wait(run_time=start_time_), AnimationGroup(Write(pxy[cell_id][0]),
                                                                             Write(pxy[cell_id][2]),
                                                                             Write(pxy[cell_id][4])),
                                  x_ghost.animate.move_to(pxy[cell_id][3]),
                                  y_ghost.animate.move_to(pxy[cell_id][1])
                                  )
                       for cell_id, start_time_, x_ghost, y_ghost in
                       zip(range(1, len(x_coords) - 1), start_times, x_ghosts, y_ghosts_of_ghosts)]
        self.wait(15)
        self.play(*point_anims, run_time=3)  #(1:44)

        ################################# Probability formula
        prob_formula_font = 40
        prob_formula_color = YELLOW
        prob_formula = MathTex(r"p(", r"y", r"\vert", r"x", r")", r" = \begin{cases} \frac{1}{1+e^{-(ax + b)}}, &y = 1 \\ 1 - \frac{1}{1+e^{-(ax + b)}}, &y = 0 \end{cases}",
        font_size=prob_formula_font, color=prob_formula_color).shift(DOWN * 1.25 + RIGHT * 1.5)
        self.wait(1.5)
        self.play(Write(prob_formula[0:5]), run_time=1.5)   #(1:47)

        ################################# Draw circles around x and y
        radius = 0.16
        stroke_width = 3
        circle_color = WHITE
        x_circle = draw_circle_around_point(p=prob_formula[3], radius=radius, stroke_width=stroke_width, color=circle_color)
        self.play(Write(x_circle), run_time=0.75)
        self.play(Unwrite(x_circle), run_time=0.75)  #(1:48.5)
        y_circle = draw_circle_around_point(p=prob_formula[1], radius=radius, stroke_width=stroke_width, color=circle_color)
        self.wait(2)
        self.play(Write(y_circle), run_time=0.75)
        self.play(Unwrite(y_circle), run_time=0.75)  #(1:52)

        ################################# Write the formula
        last_idx = 18
        self.wait(7.5)
        self.play(Write(prob_formula[5][:last_idx]), run_time=1.5)  #(2:01)
        self.wait(4.5)
        self.play(Write(prob_formula[5][last_idx:]), run_time=1.5)  #(2:07)

        ################################# Aggregated formula
        agg_formula = MathTex(r"\max", r" p(y_1 \vert x_1)", r" * ", r"p(y_2 \vert x_2)", r" * ", r"\dots", r" * ", r"p(y_m \vert x_m)",
                              r"\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\rightarrow", r"a, b",
                              font_size=prob_formula_font, color=prob_formula_color).shift(DOWN * 3.1)
        agg_formula.set_color_by_tex("max", BLUE)
        agg_formula.set_color_by_tex("*", BLUE)
        agg_formula.set_color_by_tex("arrow", BLUE)
        agg_formula.set_color_by_tex("a, b", BLUE)

        # Write terms
        self.wait(3.5)
        self.play(Write(agg_formula[1]), Write(agg_formula[3]), Write(agg_formula[5]), Write(agg_formula[7]),
                  run_time=1.5)  #(2:12)

        # Write multiplication signs
        self.wait(6.5)
        self.play(Write(agg_formula[2]), Write(agg_formula[4]), Write(agg_formula[6]), run_time=1.5)  #(2:20)

        # Write max
        self.wait(2)
        self.play(Write(agg_formula[0]), run_time=1.5)  #max
        self.play(Write(agg_formula[8]), run_time=1.5)  #arrow
        self.play(Write(agg_formula[9]), run_time=1.5)  #a, b;
        self.wait(0.5)  #(2:27)

        ################################# Remove all
        all_objects = [m for m in self.mobjects if m is not mle_text]
        anims = [FadeOut(obj) for obj in all_objects]
        self.wait(9)
        self.play(*anims)

        ################################# Write the final formula
        final_formula = MathTex(r"\min", r" - \frac{1}{m} \sum_{i=1}^m \left[ y_i \log (\frac{1}{1+e^{-(ax_i + b)}}) + (1 - y_i) \log (1 - \frac{1}{1+e^{-(ax_i + b)}}) \right]",
                              font_size=prob_formula_font, color=prob_formula_color)
        final_formula.set_color_by_tex("min", BLUE)
        self.wait(1)
        self.play(Write(final_formula))
        self.wait(1)  #(2:39)

        self.wait(14.2)
        self.play(FadeOut(final_formula), FadeOut(mle_text))


from manim import *

from manim import *

class Thumbnail(ThreeDScene):
    def construct(self):
        # load image
        image = ImageMobject("data/frame_black.png").scale(1)
        self.add(image)

        # add text
        shift = 3
        base_text = "Solving Logistic Regression"
        depth = 10
        step = 0.05

        texts = [
            Text(base_text, font_size=80, color=BLACK)
            .shift(UP * shift + (i * step) * DOWN + (i * step) * RIGHT)
            .set_opacity(0.5 - (i * 0.05))
            for i in range(depth)
        ]
        texts.append(Text(base_text, font_size=80, color=YELLOW).shift(UP * shift))
        text = VGroup(*texts)

        # add 3D text
        self.add(text)

