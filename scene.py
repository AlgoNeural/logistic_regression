# This software uses the community version of the manim animation library for python
# https://github.com/ManimCommunity/manim
# The manim license can be found in manim-license.txt.
import numpy as np
from manim import *
from scipy import stats
from sklearn.linear_model import LogisticRegression

COLOR_SUCCESS = GREEN
COLOR_FAIL = RED
COLOR_UNKNOWN = YELLOW
COLOR_WRONG_LABEL = PURPLE


stroke_width_line05 = 0.9

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


def get_axes(x_range, y_range, x_length, y_length, scale, arrow_tip_scale=None):
    axis_config = {
        "include_ticks": False,
        "color": BLUE
    }
    if arrow_tip_scale is not None:
        axis_config["tip_length"] = arrow_tip_scale
        axis_config["tip_width"] = arrow_tip_scale
    return Axes(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length,
                axis_config=axis_config,
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


def get_func_graph(a, b, x_range, y_range, x_length, y_length, scale, type='linear'):
    arrow_tip_scale = 0.12
    ax = get_axes(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length, scale=scale,
                  arrow_tip_scale=arrow_tip_scale)
    if type == 'linear':
        my_func = lambda t: a * t + b
    elif type == 'sigmoid':
        my_func = lambda t: 1 / (1 + np.exp(-(a * t + b)))
    else:
        raise ValueError('Wrong type!')
    linear_fcn = ParametricFunction(lambda t: ax.c2p(t, my_func(t)), t_range=[x_range[0] * 0.8, x_range[1] * 0.8],
                                    stroke_width=2)
    return VGroup(ax, linear_fcn), {"ax": ax, "fcn": linear_fcn}


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


def get_axes(x_range, y_range, x_length, y_length, scale, arrow_tip_scale=None):
    axis_config = {
        "include_ticks": False,
        "color": BLUE,
        "stroke_width": 5,
    }
    if arrow_tip_scale is not None:
        axis_config["tip_length"] = arrow_tip_scale
        axis_config["tip_width"] = arrow_tip_scale
    return Axes(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length,
                axis_config=axis_config,
                ).scale(scale)


class Thumbnail(ThreeDScene):

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def get_sigmoid_curve(self, ax, x_range, horizontal_squish=1):
        return ParametricFunction(lambda t: ax.c2p(t, self.sigmoid(t * horizontal_squish)), t_range=x_range,
                                  stroke_width=13)

    def get_logistic_func(self, x_range, y_range, x_length, y_length, scale, font_size, formula_shift_up,
                          formula_shift_left):
        arrow_tip_scale = 0.15
        ax = get_axes(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length, scale=scale,
                      arrow_tip_scale=arrow_tip_scale)
        logistic_fcn = self.get_sigmoid_curve(ax=ax, x_range=x_range)
        # equation = MathTex(r'f(x) = \frac{1}{1 + e^{-x}}', font_size=font_size). \
        #     move_to(ax.get_center() + LEFT * formula_shift_left + UP * formula_shift_up).set_color(YELLOW)

        return VGroup(ax, logistic_fcn), {"ax": ax, "logistic_fcn": logistic_fcn}

    def get_linear_func(self, a, b, x_range, y_range, x_length, y_length, scale):
        arrow_tip_scale = 0.26
        ax = get_axes(x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length, scale=scale,
                      arrow_tip_scale=arrow_tip_scale)
        linear_fcn = ParametricFunction(lambda t: ax.c2p(t, a * t + b), t_range=x_range)
        return VGroup(ax, linear_fcn), {"ax": ax, "fcn": linear_fcn}

    def get_human(self):
        return ImageMobject("data/penguin.png").scale(0.15)

    def get_puffin(self):
        return ImageMobject("data/puffin.png").scale(0.15)

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
        y_range = [-0.2, 1.1]
        x_length = 5
        y_length = 2.1
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
        # pierre_img = ImageMobject("data/Pierre_Francois_Verhulst.jpeg").scale(1.3).shift(RIGHT * 4.5)

        # Population growth
        people, people_list = self.get_people()
        # people.move_to(pierre_img.get_center() + DOWN * 4).scale(0.7)
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
        logistic_regression_text = Tex(r'\begin{minipage}{15cm}Logistic Regression\end{minipage}', font_size=100)

        data_x = np.array([-4, -2.3, -1.5, 1.1, 2, 3, 3.7])
        data_y = np.array([0, 0, 0, 1, 1, 1, 1])
        res = stats.linregress(data_x, data_y)
        linear_fcn, linear_fcn_dict = self.get_linear_func(res.slope, res.intercept, x_range, y_range=[-1.5, 1.5],
                                                           x_length=x_length,
                                                           y_length=2.5, scale=1.4)
        # linear_fcn.shift(LEFT + DOWN * 0.75)
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
        self.play(Create(logistic_graph_dict["ax"]), run_time=0.1)
        self.play(Write(logistic_graph_dict["logistic_fcn"]), run_time=0.1)

        logistic_graph_colorful.scale(1.2).shift(DOWN * 1)
        logistic_regression_text.next_to(logistic_graph_colorful, UP * 3.5)
        self.play(Transform(logistic_graph, logistic_graph_colorful), run_time=0.1)
        # self.play(FadeIn(pierre_img), run_time=0.1)
        self.play(Write(logistic_regression_text), run_time=0.1)

        n_penguins = 5
        penguin_x = [-4 + i * 0.6 for i in range(n_penguins)]
        penguin_y = [self.sigmoid(x * 1) for x in penguin_x]
        ax = logistic_graph_dict["ax"]
        penguin_pos = [ax.c2p(x, y) for x, y in zip(penguin_x, penguin_y)]
        penguins = [self.get_human().move_to(penguin_pos[i]).shift(UP * 0.6) for i in range(n_penguins)]
        self.play(*[FadeIn(p) for p in penguins])

        n_puffins = 5
        puffin_x = [4 - i * 0.6 for i in range(n_puffins)]
        puffin_y = [self.sigmoid(x * 1) for x in puffin_x]
        ax = logistic_graph_dict["ax"]
        puffin_pos = [ax.c2p(x, y) for x, y in zip(puffin_x, puffin_y)]
        puffins = [self.get_puffin().scale(1.1).move_to(puffin_pos[i]).shift(UP * 0.45) for i in range(n_puffins)]
        self.play(*[FadeIn(p) for p in puffins])

        self.wait(0.2)
