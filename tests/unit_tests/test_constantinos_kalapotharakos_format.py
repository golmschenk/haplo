import pytest

from haplo.internal.constantinos_kalapotharakos_format import \
    constantinos_kalapotharakos_format_record_generator_from_file_contents, ConstantinosKalapotharakosFormatError


def test_constantinos_kalapotharakos_format_record_generator():
    example_file_contents = b""" -0.125496240826648       3.060533215990838E-002  0.151505257392365     
   1.65013990622357        2.32804991612597       0.451147517955635     
  0.250608101710060      -7.359561018925495E-002   2.32004452858950     
   2.34352378081674        6.36995085862800       -416.488760907102     
           0
 -7.135721876204401E-002 -0.427548407950883      -0.353689894273218     
  0.876778877275623        1.76011619152630       0.621144611139376     
  0.193366343299776       0.170144348612416        2.36359014240431     
   2.57244222257307        10.3067805058740       -428.355978949089     
           1
 -0.125496240826648       3.060533215990838E-002  0.151505257392365     
   1.65013990622357        2.32804991612597       0.451147517955635     
  0.250608101710060      -7.359561018925495E-002   2.32004452858950     
   2.34352378081674        6.36995085862800       -416.488760907102     
           0"""

    generator = constantinos_kalapotharakos_format_record_generator_from_file_contents(example_file_contents,
                                                                                       elements_per_record=13)
    list_ = list(generator)
    assert len(list_) == 3
    assert list_[1][2] == pytest.approx(-0.353689894273218)


def test_constantinos_kalapotharakos_format_record_generator_raises_error_for_the_wrong_number_of_elements_found():
    example_file_contents = b""" -0.125496240826648       3.060533215990838E-002  0.151505257392365     
   1.65013990622357        2.32804991612597       0.451147517955635     
  0.250608101710060      -7.359561018925495E-002   2.32004452858950     
   2.34352378081674        6.36995085862800       -416.488760907102     
           0
 -7.135721876204401E-002 -0.427548407950883      -0.353689894273218     
  0.876778877275623        1.76011619152630"""

    generator = constantinos_kalapotharakos_format_record_generator_from_file_contents(example_file_contents,
                                                                                       elements_per_record=13)
    with pytest.raises(ConstantinosKalapotharakosFormatError):
        _ = list(generator)
