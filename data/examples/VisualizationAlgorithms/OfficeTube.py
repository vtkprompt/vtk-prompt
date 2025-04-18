#!/usr/bin/env python

# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonDataModel import vtkDataObject
from vtkmodules.vtkCommonMath import vtkRungeKutta4
from vtkmodules.vtkFiltersCore import (
    vtkStructuredGridOutlineFilter,
    vtkTubeFilter
)
from vtkmodules.vtkFiltersFlowPaths import vtkStreamTracer
from vtkmodules.vtkFiltersGeometry import vtkStructuredGridGeometryFilter
from vtkmodules.vtkIOLegacy import vtkDataSetReader
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCamera,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)


def main():
    fileName = get_program_parameters()

    colors = vtkNamedColors()
    # Set the furniture colors, matching those in the VTKTextBook.
    tableTopColor = [0.59, 0.427, 0.392]
    filingCabinetColor = [0.8, 0.8, 0.6]
    bookShelfColor = [0.8, 0.8, 0.6]
    windowColor = [0.3, 0.3, 0.5]
    colors.SetColor('TableTop', *tableTopColor)
    colors.SetColor('FilingCabinet', *filingCabinetColor)
    colors.SetColor('BookShelf', *bookShelfColor)
    colors.SetColor('WindowColor', *windowColor)

    # We read a data file that represents a CFD analysis of airflow in an office
    # (with ventilation and a burning cigarette). We force an update so that we
    # can query the output for its length, i.e., the length of the diagonal
    # of the bounding box. This is useful for normalizing the data.
    reader = vtkDataSetReader()
    reader.SetFileName(fileName)
    reader.Update()

    # Now we will generate a single streamline in the data. We select the
    # integration order to use (RungeKutta order 4) and associate it with
    # the streamer. The start position is the position in world space where
    # we want to begin streamline integration; and we integrate in both
    # directions. The step length is the length of the line segments that
    # make up the streamline (i.e., related to display). The
    # IntegrationStepLength specifies the integration step length as a
    # fraction of the cell size that the streamline is in.
    integ = vtkRungeKutta4()

    streamer = vtkStreamTracer()
    streamer.SetInputConnection(reader.GetOutputPort())
    streamer.SetStartPosition(0.1, 2.1, 0.5)
    streamer.SetMaximumPropagation(500)
    streamer.SetInitialIntegrationStep(0.05)
    streamer.SetIntegrationDirectionToBoth()
    streamer.SetIntegrator(integ)

    # The tube is wrapped around the generated streamline. By varying the radius
    # by the inverse of vector magnitude, we are creating a tube whose radius is
    # proportional to mass flux (in incompressible flow).
    streamTube = vtkTubeFilter()
    streamTube.SetInputConnection(streamer.GetOutputPort())
    streamTube.SetInputArrayToProcess(1, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, 'vectors')
    streamTube.SetRadius(0.02)
    streamTube.SetNumberOfSides(12)
    streamTube.SetVaryRadiusToVaryRadiusByVector()

    mapStreamTube = vtkPolyDataMapper()
    mapStreamTube.SetInputConnection(streamTube.GetOutputPort())
    mapStreamTube.SetScalarRange(reader.GetOutput().GetPointData().GetScalars().GetRange())

    streamTubeActor = vtkActor()
    streamTubeActor.SetMapper(mapStreamTube)
    streamTubeActor.GetProperty().BackfaceCullingOn()

    # Create the scene.
    # We generate a whole bunch of planes which correspond to
    # the geometry in the analysis; tables, bookshelves and so on.
    table1 = vtkStructuredGridGeometryFilter()
    table1.SetInputData(reader.GetStructuredGridOutput())
    table1.SetExtent(11, 15, 7, 9, 8, 8)
    mapTable1 = vtkPolyDataMapper()
    mapTable1.SetInputConnection(table1.GetOutputPort())
    mapTable1.ScalarVisibilityOff()
    table1Actor = vtkActor()
    table1Actor.SetMapper(mapTable1)
    table1Actor.GetProperty().SetColor(colors.GetColor3d('TableTop'))

    table2 = vtkStructuredGridGeometryFilter()
    table2.SetInputData(reader.GetStructuredGridOutput())
    table2.SetExtent(11, 15, 10, 12, 8, 8)
    mapTable2 = vtkPolyDataMapper()
    mapTable2.SetInputConnection(table2.GetOutputPort())
    mapTable2.ScalarVisibilityOff()
    table2Actor = vtkActor()
    table2Actor.SetMapper(mapTable2)
    table2Actor.GetProperty().SetColor(colors.GetColor3d('TableTop'))

    FilingCabinet1 = vtkStructuredGridGeometryFilter()
    FilingCabinet1.SetInputData(reader.GetStructuredGridOutput())
    FilingCabinet1.SetExtent(15, 15, 7, 9, 0, 8)
    mapFilingCabinet1 = vtkPolyDataMapper()
    mapFilingCabinet1.SetInputConnection(FilingCabinet1.GetOutputPort())
    mapFilingCabinet1.ScalarVisibilityOff()
    FilingCabinet1Actor = vtkActor()
    FilingCabinet1Actor.SetMapper(mapFilingCabinet1)
    FilingCabinet1Actor.GetProperty().SetColor(colors.GetColor3d('FilingCabinet'))

    FilingCabinet2 = vtkStructuredGridGeometryFilter()
    FilingCabinet2.SetInputData(reader.GetStructuredGridOutput())
    FilingCabinet2.SetExtent(15, 15, 10, 12, 0, 8)
    mapFilingCabinet2 = vtkPolyDataMapper()
    mapFilingCabinet2.SetInputConnection(FilingCabinet2.GetOutputPort())
    mapFilingCabinet2.ScalarVisibilityOff()
    FilingCabinet2Actor = vtkActor()
    FilingCabinet2Actor.SetMapper(mapFilingCabinet2)
    FilingCabinet2Actor.GetProperty().SetColor(colors.GetColor3d('FilingCabinet'))

    bookshelf1Top = vtkStructuredGridGeometryFilter()
    bookshelf1Top.SetInputData(reader.GetStructuredGridOutput())
    bookshelf1Top.SetExtent(13, 13, 0, 4, 0, 11)
    mapBookshelf1Top = vtkPolyDataMapper()
    mapBookshelf1Top.SetInputConnection(bookshelf1Top.GetOutputPort())
    mapBookshelf1Top.ScalarVisibilityOff()
    bookshelf1TopActor = vtkActor()
    bookshelf1TopActor.SetMapper(mapBookshelf1Top)
    bookshelf1TopActor.GetProperty().SetColor(colors.GetColor3d('BookShelf'))

    bookshelf1Bottom = vtkStructuredGridGeometryFilter()
    bookshelf1Bottom.SetInputData(reader.GetStructuredGridOutput())
    bookshelf1Bottom.SetExtent(20, 20, 0, 4, 0, 11)
    mapBookshelf1Bottom = vtkPolyDataMapper()
    mapBookshelf1Bottom.SetInputConnection(bookshelf1Bottom.GetOutputPort())
    mapBookshelf1Bottom.ScalarVisibilityOff()
    bookshelf1BottomActor = vtkActor()
    bookshelf1BottomActor.SetMapper(mapBookshelf1Bottom)
    bookshelf1BottomActor.GetProperty().SetColor(colors.GetColor3d('BookShelf'))

    bookshelf1Front = vtkStructuredGridGeometryFilter()
    bookshelf1Front.SetInputData(reader.GetStructuredGridOutput())
    bookshelf1Front.SetExtent(13, 20, 0, 0, 0, 11)
    mapBookshelf1Front = vtkPolyDataMapper()
    mapBookshelf1Front.SetInputConnection(bookshelf1Front.GetOutputPort())
    mapBookshelf1Front.ScalarVisibilityOff()
    bookshelf1FrontActor = vtkActor()
    bookshelf1FrontActor.SetMapper(mapBookshelf1Front)
    bookshelf1FrontActor.GetProperty().SetColor(colors.GetColor3d('BookShelf'))

    bookshelf1Back = vtkStructuredGridGeometryFilter()
    bookshelf1Back.SetInputData(reader.GetStructuredGridOutput())
    bookshelf1Back.SetExtent(13, 20, 4, 4, 0, 11)
    mapBookshelf1Back = vtkPolyDataMapper()
    mapBookshelf1Back.SetInputConnection(bookshelf1Back.GetOutputPort())
    mapBookshelf1Back.ScalarVisibilityOff()
    bookshelf1BackActor = vtkActor()
    bookshelf1BackActor.SetMapper(mapBookshelf1Back)
    bookshelf1BackActor.GetProperty().SetColor(colors.GetColor3d('BookShelf'))

    bookshelf1LHS = vtkStructuredGridGeometryFilter()
    bookshelf1LHS.SetInputData(reader.GetStructuredGridOutput())
    bookshelf1LHS.SetExtent(13, 20, 0, 4, 0, 0)
    mapBookshelf1LHS = vtkPolyDataMapper()
    mapBookshelf1LHS.SetInputConnection(bookshelf1LHS.GetOutputPort())
    mapBookshelf1LHS.ScalarVisibilityOff()
    bookshelf1LHSActor = vtkActor()
    bookshelf1LHSActor.SetMapper(mapBookshelf1LHS)
    bookshelf1LHSActor.GetProperty().SetColor(colors.GetColor3d('BookShelf'))

    bookshelf1RHS = vtkStructuredGridGeometryFilter()
    bookshelf1RHS.SetInputData(reader.GetStructuredGridOutput())
    bookshelf1RHS.SetExtent(13, 20, 0, 4, 11, 11)
    mapBookshelf1RHS = vtkPolyDataMapper()
    mapBookshelf1RHS.SetInputConnection(bookshelf1RHS.GetOutputPort())
    mapBookshelf1RHS.ScalarVisibilityOff()
    bookshelf1RHSActor = vtkActor()
    bookshelf1RHSActor.SetMapper(mapBookshelf1RHS)
    bookshelf1RHSActor.GetProperty().SetColor(colors.GetColor3d('BookShelf'))

    bookshelf2Top = vtkStructuredGridGeometryFilter()
    bookshelf2Top.SetInputData(reader.GetStructuredGridOutput())
    bookshelf2Top.SetExtent(13, 13, 15, 19, 0, 11)
    mapBookshelf2Top = vtkPolyDataMapper()
    mapBookshelf2Top.SetInputConnection(bookshelf2Top.GetOutputPort())
    mapBookshelf2Top.ScalarVisibilityOff()
    bookshelf2TopActor = vtkActor()
    bookshelf2TopActor.SetMapper(mapBookshelf2Top)
    bookshelf2TopActor.GetProperty().SetColor(colors.GetColor3d('BookShelf'))

    bookshelf2Bottom = vtkStructuredGridGeometryFilter()
    bookshelf2Bottom.SetInputData(reader.GetStructuredGridOutput())
    bookshelf2Bottom.SetExtent(20, 20, 15, 19, 0, 11)
    mapBookshelf2Bottom = vtkPolyDataMapper()
    mapBookshelf2Bottom.SetInputConnection(bookshelf2Bottom.GetOutputPort())
    mapBookshelf2Bottom.ScalarVisibilityOff()
    bookshelf2BottomActor = vtkActor()
    bookshelf2BottomActor.SetMapper(mapBookshelf2Bottom)
    bookshelf2BottomActor.GetProperty().SetColor(colors.GetColor3d('BookShelf'))

    bookshelf2Front = vtkStructuredGridGeometryFilter()
    bookshelf2Front.SetInputData(reader.GetStructuredGridOutput())
    bookshelf2Front.SetExtent(13, 20, 15, 15, 0, 11)
    mapBookshelf2Front = vtkPolyDataMapper()
    mapBookshelf2Front.SetInputConnection(bookshelf2Front.GetOutputPort())
    mapBookshelf2Front.ScalarVisibilityOff()
    bookshelf2FrontActor = vtkActor()
    bookshelf2FrontActor.SetMapper(mapBookshelf2Front)
    bookshelf2FrontActor.GetProperty().SetColor(colors.GetColor3d('BookShelf'))

    bookshelf2Back = vtkStructuredGridGeometryFilter()
    bookshelf2Back.SetInputData(reader.GetStructuredGridOutput())
    bookshelf2Back.SetExtent(13, 20, 19, 19, 0, 11)
    mapBookshelf2Back = vtkPolyDataMapper()
    mapBookshelf2Back.SetInputConnection(bookshelf2Back.GetOutputPort())
    mapBookshelf2Back.ScalarVisibilityOff()
    bookshelf2BackActor = vtkActor()
    bookshelf2BackActor.SetMapper(mapBookshelf2Back)
    bookshelf2BackActor.GetProperty().SetColor(colors.GetColor3d('BookShelf'))

    bookshelf2LHS = vtkStructuredGridGeometryFilter()
    bookshelf2LHS.SetInputData(reader.GetStructuredGridOutput())
    bookshelf2LHS.SetExtent(13, 20, 15, 19, 0, 0)
    mapBookshelf2LHS = vtkPolyDataMapper()
    mapBookshelf2LHS.SetInputConnection(bookshelf2LHS.GetOutputPort())
    mapBookshelf2LHS.ScalarVisibilityOff()
    bookshelf2LHSActor = vtkActor()
    bookshelf2LHSActor.SetMapper(mapBookshelf2LHS)
    bookshelf2LHSActor.GetProperty().SetColor(colors.GetColor3d('BookShelf'))

    bookshelf2RHS = vtkStructuredGridGeometryFilter()
    bookshelf2RHS.SetInputData(reader.GetStructuredGridOutput())
    bookshelf2RHS.SetExtent(13, 20, 15, 19, 11, 11)
    mapBookshelf2RHS = vtkPolyDataMapper()
    mapBookshelf2RHS.SetInputConnection(bookshelf2RHS.GetOutputPort())
    mapBookshelf2RHS.ScalarVisibilityOff()
    bookshelf2RHSActor = vtkActor()
    bookshelf2RHSActor.SetMapper(mapBookshelf2RHS)
    bookshelf2RHSActor.GetProperty().SetColor(colors.GetColor3d('BookShelf'))

    window = vtkStructuredGridGeometryFilter()
    window.SetInputData(reader.GetStructuredGridOutput())
    window.SetExtent(20, 20, 6, 13, 10, 13)
    mapWindow = vtkPolyDataMapper()
    mapWindow.SetInputConnection(window.GetOutputPort())
    mapWindow.ScalarVisibilityOff()
    windowActor = vtkActor()
    windowActor.SetMapper(mapWindow)
    windowActor.GetProperty().SetColor(colors.GetColor3d('WindowColor'))

    outlet = vtkStructuredGridGeometryFilter()
    outlet.SetInputData(reader.GetStructuredGridOutput())
    outlet.SetExtent(0, 0, 9, 10, 14, 16)
    mapOutlet = vtkPolyDataMapper()
    mapOutlet.SetInputConnection(outlet.GetOutputPort())
    mapOutlet.ScalarVisibilityOff()
    outletActor = vtkActor()
    outletActor.SetMapper(mapOutlet)
    outletActor.GetProperty().SetColor(colors.GetColor3d('lamp_black'))

    inlet = vtkStructuredGridGeometryFilter()
    inlet.SetInputData(reader.GetStructuredGridOutput())
    inlet.SetExtent(0, 0, 9, 10, 0, 6)
    mapInlet = vtkPolyDataMapper()
    mapInlet.SetInputConnection(inlet.GetOutputPort())
    mapInlet.ScalarVisibilityOff()
    inletActor = vtkActor()
    inletActor.SetMapper(mapInlet)
    inletActor.GetProperty().SetColor(colors.GetColor3d('lamp_black'))

    outline = vtkStructuredGridOutlineFilter()
    outline.SetInputData(reader.GetStructuredGridOutput())
    mapOutline = vtkPolyDataMapper()
    mapOutline.SetInputConnection(outline.GetOutputPort())
    outlineActor = vtkActor()
    outlineActor.SetMapper(mapOutline)
    outlineActor.GetProperty().SetColor(colors.GetColor3d('Black'))

    # Create the rendering window, renderer, and interactive renderer.
    ren = vtkRenderer()
    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Add the remaining actors to the renderer, set the background and size.
    ren.AddActor(table1Actor)
    ren.AddActor(table2Actor)
    ren.AddActor(FilingCabinet1Actor)
    ren.AddActor(FilingCabinet2Actor)
    ren.AddActor(bookshelf1TopActor)
    ren.AddActor(bookshelf1BottomActor)
    ren.AddActor(bookshelf1FrontActor)
    ren.AddActor(bookshelf1BackActor)
    ren.AddActor(bookshelf1LHSActor)
    ren.AddActor(bookshelf1RHSActor)
    ren.AddActor(bookshelf2TopActor)
    ren.AddActor(bookshelf2BottomActor)
    ren.AddActor(bookshelf2FrontActor)
    ren.AddActor(bookshelf2BackActor)
    ren.AddActor(bookshelf2LHSActor)
    ren.AddActor(bookshelf2RHSActor)
    ren.AddActor(windowActor)
    ren.AddActor(outletActor)
    ren.AddActor(inletActor)
    ren.AddActor(outlineActor)
    ren.AddActor(streamTubeActor)

    ren.SetBackground(colors.GetColor3d('SlateGray'))

    aCamera = vtkCamera()
    aCamera.SetClippingRange(0.726079, 36.3039)
    aCamera.SetFocalPoint(2.43584, 2.15046, 1.11104)
    aCamera.SetPosition(-4.76183, -10.4426, 3.17203)
    aCamera.ComputeViewPlaneNormal()
    aCamera.SetViewUp(0.0511273, 0.132773, 0.989827)
    aCamera.SetViewAngle(18.604)
    aCamera.Zoom(1.2)

    ren.SetActiveCamera(aCamera)

    renWin.SetSize(640, 400)
    renWin.SetWindowName('OfficeTube')

    iren.Initialize()
    iren.Start()


def get_program_parameters():
    import argparse
    description = 'The stream polygon. Sweeping a polygon to form a tube..'
    epilogue = '''
   '''
    parser = argparse.ArgumentParser(description=description, epilog=epilogue,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('fileName', help='office.binary.vtk')
    args = parser.parse_args()
    return args.fileName


if __name__ == '__main__':
    main()
