program example_using_infer_from_parameters_to_phase_amplitudes
    use, intrinsic :: iso_c_binding, only : c_ptr, c_float, c_int, c_loc
    implicit none
    interface
        subroutine infer_from_parameters_to_phase_amplitudes(parameters_array_pointer, phase_amplitudes_array_pointer) &
                bind(c, name = 'infer_from_parameters_to_phase_amplitudes')
            import :: c_ptr
            type(c_ptr), intent(in), value :: parameters_array_pointer
            type(c_ptr), intent(in), value :: phase_amplitudes_array_pointer
        end subroutine
    end interface
    real(c_float), dimension(:), allocatable, target :: parameters_array(:)
    real(c_float), dimension(:), allocatable, target :: phase_amplitudes_array(:)
    type(c_ptr) :: parameters_array_pointer_
    type(c_ptr) :: phase_amplitudes_array_pointer_
    integer, parameter :: number_of_parameters = 11
    integer, parameter :: number_of_phase_amplitudes = 64
    allocate(parameters_array(number_of_parameters))
    allocate(phase_amplitudes_array(number_of_phase_amplitudes))
    parameters_array_pointer_ = c_loc(parameters_array(1))
    phase_amplitudes_array_pointer_ = c_loc(phase_amplitudes_array(1))
    parameters_array = (/ -0.137349282472716, 4.651922986569446E-002, -0.126309026142708, 2.57614122691645, &
            3.94358482944553, 0.303202923979724, 0.132341360556433, 0.304479697430865, 0.758863131388038, &
            3.84855473811096, 2.77055893884855 /)
    write(*, *) "Parameters:"
    write(*, *) parameters_array
    call infer_from_parameters_to_phase_amplitudes(parameters_array_pointer_, phase_amplitudes_array_pointer_)
    write(*, *) "Phase amplitudes:"
    write(*, *) phase_amplitudes_array
end program example_using_infer_from_parameters_to_phase_amplitudes
