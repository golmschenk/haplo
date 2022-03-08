program example_using_infer_from_parameters_to_phase_amplitudes
    use, intrinsic :: iso_c_binding, only : c_ptr, c_float, c_int, c_loc
    implicit none
    interface
        subroutine infer_from_parameters_to_phase_amplitudes(input_array_pointer, output_array_pointer) &
                bind(c, name = 'infer_from_parameters_to_phase_amplitudes')
            import :: c_ptr
            type(c_ptr), intent(in), value :: input_array_pointer
            type(c_ptr), intent(in), value :: output_array_pointer
        end subroutine
    end interface
    real(c_float), dimension(:), allocatable, target :: input_array(:)
    real(c_float), dimension(:), allocatable, target :: output_array(:)
    type(c_ptr) :: input_array_pointer_
    type(c_ptr) :: output_array_pointer_
    integer, parameter :: number_of_parameters = 11
    integer, parameter :: number_of_phase_amplitudes = 64
    allocate(input_array(number_of_parameters))
    allocate(output_array(number_of_phase_amplitudes))
    input_array_pointer_ = c_loc(input_array(1))
    output_array_pointer_ = c_loc(output_array(1))
    input_array = (/ -0.137349282472716, 4.651922986569446E-002, -0.126309026142708, 2.57614122691645, &
            3.94358482944553, 0.303202923979724, 0.132341360556433, 0.304479697430865, 0.758863131388038, &
            3.84855473811096, 2.77055893884855 /)
    write(*, *) "Parameters:"
    write(*, *) input_array
    call infer_from_parameters_to_phase_amplitudes(input_array_pointer_, output_array_pointer_)
    write(*, *) "Phase amplitudes:"
    write(*, *) output_array
end program example_using_infer_from_parameters_to_phase_amplitudes
