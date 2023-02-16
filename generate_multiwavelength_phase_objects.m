close all;clear all

I0 = 1; % incident intensity
d = 0.5e-7; % pixel spacing
lam1 = 505e-9; % wavelength
lam2 = 488e-9; % wavelength
alpha_water = 2e-1; % absorption coefficient of water at 633 nm
W_nucleus = 0.87; % water content of nucleus
W_cytoplasm = 0.44; % water content of cytoplasm
W_ribosome = 1;
W_mitochondria = 1;

n_medium1 = 1.33; % index of refraction of medium
n_medium2 = 1.35; % index of refraction of mediumrx_cytoplasm

% refractive index parameters
n_cytoplasm0 = 1.36; % index of refraction of cytoplasm
n_nucleus0 = 1.5; % index of refraction of nucleus
n_ribosome0 = 1.55;
n_mitochondria0 = 1.34;

n_cytoplasm_std = 0.005;
n_nucleus_std = 0.005;
n_ribosome_std = 0.01;
n_mitochondria_std = 0.005;

% shape parameters
rx_cytoplasm0 = 1e-6*5;
ry_cytoplasm0 = 1e-6*5;
rz_cytoplasm0 = 1e-6*5;
r_nucleus0 = 1e-6*1.2; % radius of nucleus ( 2.97 microns )
r_ribosome0 = 1e-6*0.25;
rx_mitochondria0 = 1e-6*2;
ry_mitochondria0 = 1e-6*3.5;
rz_mitochondria0 = 1e-6*2;

rx_cytoplasm_std = 1e-6*5/15;
ry_cytoplasm_std = 1e-6*5/15;
rz_cytoplasm_std = 1e-6*5/15;
r_nucleus_std = 1e-6*1.25/15;
r_ribosome_std = 1e-6*0.25/15;
rx_mitochondria_std = 1e-6*2.5/15;
ry_mitochondria_std = 1e-6*4/15;
rz_mitochondria_std = 1e-6*2.5/15;

M = 256; N = 256; % image dimensions
n = ( 0:N-1 )-N/2; m = ( 0:M-1 )-M/2; % pixel grid
x = d*n; y = d*m; [ X,Y ] = meshgrid( x,y ); % spatial grid

nRealizations = 128;
for iRealization = 1:nRealizations
    n = ( 0:N-1 )-N/2; m = ( 0:M-1 )-M/2; % pixel grid
    x = d*n; y = d*m; [ X,Y ] = meshgrid( x,y ); % spatial grid
    rx_cytoplasm = rx_cytoplasm0-rx_cytoplasm_std*randn( 1,1 );
    ry_cytoplasm = rx_cytoplasm;
    rz_cytoplasm = rz_cytoplasm0-rz_cytoplasm_std*randn( 1,1 );
    r_nucleus = r_nucleus0-r_nucleus_std*randn( 1,1 );
    r_ribosome = r_ribosome0-r_ribosome_std*randn( 1,1 );
    rx_mitochondria = rx_mitochondria0-rx_mitochondria_std*randn( 1,1 );
    ry_mitochondria = ry_mitochondria0-ry_mitochondria_std*randn( 1,1 );
    rz_mitochondria = rz_mitochondria0-rz_mitochondria_std*randn( 1,1 );

    x_mitochondria_std = rx_cytoplasm/10;
    y_mitochondria_std = rx_cytoplasm/10;
    x_nucleus_std = rx_mitochondria/10;
    y_nucleus_std = rx_mitochondria/10;

    n_cytoplasm = n_cytoplasm0-n_cytoplasm_std*randn( 1,1 );
    n_nucleus = n_nucleus0-n_nucleus_std*randn( 1,1 );
    n_mitochondria = n_mitochondria0-n_mitochondria_std*randn( 1,1 );
    n_ribosome = n_ribosome0-n_ribosome_std*randn( 1,1 );

    % initialize in memory
    cytoplasm_phase1 = zeros( length( y ),length( x ) );
    nucleus_phase1 = zeros( length( y ),length( x ) );
    ribosome_phase1 = zeros( length( y ),length( x ) );
    mitochondria_phase1 = zeros( length( y ),length( x ) );

    opd_cytoplasm1 = zeros( length( y ),length( x ) );
    opd_nucleus1 = zeros( length( y ),length( x ) );
    opd_ribosome1 = zeros( length( y ),length( x ) );
    opd_mitochondria1 = zeros( length( y ),length( x ) );

    absorption_cytoplasm1 = zeros( length( y ),length( x ) );
    absorption_nucleus1 = zeros( length( y ),length( x ) );
    absorption_ribosome1 = zeros( length( y ),length( x ) );
    absorption_mitochondria1 = zeros( length( y ),length( x ) );
    absorption1 = zeros( length( y ),length( x ) );

    cytoplasm_phase2 = zeros( length( y ),length( x ) );
    nucleus_phase2 = zeros( length( y ),length( x ) );
    ribosome_phase2 = zeros( length( y ),length( x ) );
    mitochondria_phase2 = zeros( length( y ),length( x ) );

    opd_cytoplasm2 = zeros( length( y ),length( x ) );
    opd_nucleus2 = zeros( length( y ),length( x ) );
    opd_ribosome2 = zeros( length( y ),length( x ) );
    opd_mitochondria2 = zeros( length( y ),length( x ) );

    absorption_cytoplasm2 = zeros( length( y ),length( x ) );
    absorption_nucleus2 = zeros( length( y ),length( x ) );
    absorption_ribosome2 = zeros( length( y ),length( x ) );
    absorption_mitochondria2 = zeros( length( y ),length( x ) );
    absorption2 = zeros( length( y ),length( x ) );

    hz = zeros( length( y ),length( x ) );
    z = linspace( -6e-6,6e-6,501 );
    nc1 = n_medium1*ones( length( y ),length( x ),length( z ) );
    nc2 = n_medium2*ones( length( y ),length( x ),length( z ) );

    x_cytoplasm = x( end/2 );
    y_cytoplasm = y( end/2 );

    % determine pixels containing cell realization
    xx = ( round( ( x_cytoplasm-rx_cytoplasm )/d ):round( ( x_cytoplasm+rx_cytoplasm )/d ) )*d;
    yy = ( round( ( y_cytoplasm-rx_cytoplasm )/d ):round( ( y_cytoplasm+rx_cytoplasm )/d ) )*d;
    [ XX,YY ] = meshgrid( xx,yy );
    xind = zeros( 1,length( xx ) );
    yind = zeros( 1,length( yy ) );
    for ix = 1:length( xx )
        [ ~,xind( ix ) ] = min( abs( n-xx( ix )/d ) );
    end
    for iy = 1:length( yy )
        [ ~,yind( iy ) ] = min( abs( m-yy( iy )/d ) );
    end

    orientation = -pi+2*pi*rand( 1 );

    z_cytoplasm = rz_cytoplasm^2*( 1-( ( XX-x_cytoplasm )/rx_cytoplasm ).^2-( ( YY-y_cytoplasm )/ry_cytoplasm ).^2 );
    z_cytoplasm( z_cytoplasm<0 ) = 0;

    [I,J] = find( z_cytoplasm==max( z_cytoplasm( : ) ) );

    z_cytoplasm = circshift( z_cytoplasm,round( size( z_cytoplasm,1 )/2 )-I,1 );
    z_cytoplasm = circshift( z_cytoplasm,round( size( z_cytoplasm,2 )/2 )-J,2 );
    z_cytoplasm = imrotate( z_cytoplasm,180/pi*orientation,'crop' );
    z_cytoplasm = circshift( z_cytoplasm,I-round( size( z_cytoplasm,1 )/2 ),1 );
    z_cytoplasm = circshift( z_cytoplasm,J-round( size( z_cytoplasm,2 )/2 ),2 );

    orientation = -pi+2*pi*rand( 1 );
    x_mitochondria = x_cytoplasm-x_mitochondria_std*randn( 1,1 );
    y_mitochondria = y_cytoplasm-y_mitochondria_std*randn( 1,1 );

    z_mitochondria = rz_mitochondria^2*( 1-( ( XX-x_mitochondria )/rx_mitochondria).^2-( ( YY-y_mitochondria )/ry_mitochondria ).^2 );
    z_mitochondria( z_mitochondria<0 ) = 0;

    [I,J] = find( z_mitochondria==max( z_mitochondria( : ) ) );

    z_mitochondria = circshift( z_mitochondria,round( size( z_mitochondria,1 )/2 )-I,1 );
    z_mitochondria = circshift( z_mitochondria,round( size( z_mitochondria,2 )/2 )-J,2 );
    z_mitochondria = imrotate( z_mitochondria,180/pi*orientation,'crop' );
    z_mitochondria = circshift( z_mitochondria,I-round( size( z_mitochondria,1 )/2 ),1 );
    z_mitochondria = circshift( z_mitochondria,J-round( size( z_mitochondria,2 )/2 ),2 );

    cytoplasm_phase1( yind,xind ) = cytoplasm_phase1( yind,xind ) + 2*pi/lam1*( n_cytoplasm-n_medium1 )*2*real( sqrt( z_cytoplasm ) );
    cytoplasm_phase2( yind,xind ) = cytoplasm_phase2( yind,xind ) + 2*pi/lam2*( n_cytoplasm-n_medium2 )*2*real( sqrt( z_cytoplasm ) );

    mitochondria_phase1( yind,xind ) = mitochondria_phase1( yind,xind ) + 2*pi/lam1*( n_mitochondria-n_cytoplasm )*2*real( sqrt( z_mitochondria ) );
    mitochondria_phase2( yind,xind ) = mitochondria_phase2( yind,xind ) + 2*pi/lam2*( n_mitochondria-n_cytoplasm )*2*real( sqrt( z_mitochondria ) );

    temp1 = real( sqrt( z_cytoplasm ) );
    temp2 = real( sqrt( z_mitochondria ) );
    nnc1 = n_medium1*ones( [ size( temp1 ) length( z ) ] );
    nnc2 = n_medium2*ones( [ size( temp2 ) length( z ) ] );
    for ii = 1:size( temp1,1 )
        for jj = 1:size( temp1,2 )
            [ ~,z1 ] = min( abs( z-temp1( ii,jj ) ) );
            [ ~,z2 ] = min( abs( z-temp2( ii,jj ) ) );
            [ ~,z3 ] = min( abs( z+temp1( ii,jj ) ) );
            [ ~,z4 ] = min( abs( z+temp2( ii,jj ) ) );
            if temp1( ii,jj ) > 0
                nnc1( ii,jj,z3:z1 ) = n_cytoplasm;
                nnc2( ii,jj,z3:z1 ) = n_cytoplasm;
            end
            if temp2( ii,jj ) > 0
                nnc1( ii,jj,z4:z2 ) = n_mitochondria;
                nnc2( ii,jj,z4:z2 ) = n_mitochondria;
            end
        end
    end

    x_nucleus = x_mitochondria-x_nucleus_std*randn( 1,1 );
    y_nucleus = y_mitochondria-y_nucleus_std*randn( 1,1 );

    % height profile of cytoplasm and nucleus
    z_nucleus = r_nucleus^2-( YY-y_nucleus ).^2-( XX-x_nucleus ).^2;
    z_nucleus( z_nucleus<0 ) = 0;

    % phase shift induced by the cytoplasm and nucleus
    nucleus_phase1( yind,xind ) = nucleus_phase1( yind,xind ) + 2*pi/lam1*( n_nucleus-n_mitochondria )*2*real( sqrt( z_nucleus ) );
    hz( yind,xind ) = real( sqrt( z_cytoplasm ) );

    nucleus_phase2( yind,xind ) = nucleus_phase2( yind,xind ) + 2*pi/lam2*( n_nucleus-n_mitochondria )*2*real( sqrt( z_nucleus ) );


    temp1 = real( sqrt( z_nucleus ) );
    for ii = 1:size( temp1,1 )
        for jj = 1:size( temp1,2 )
            [ ~,z1 ] = min( abs( z-temp1( ii,jj ) ) );
            [ ~,z2 ] = min( abs( z+temp1( ii,jj ) ) );
            if temp1( ii,jj ) > 0
                nnc1( ii,jj,z2:z1 ) = n_nucleus;
                nnc2( ii,jj,z2:z1 ) = n_nucleus;
            end
        end
    end
    nRibosome = 100;
    for jj = 1:nRibosome
        x_ribosome = x_cytoplasm-rx_cytoplasm +...
            ( x_cytoplasm+rx_cytoplasm-( x_cytoplasm-rx_cytoplasm ) ).*rand( 1,1 );

        y_ribosome  = y_cytoplasm-rx_cytoplasm +...
            ( y_cytoplasm+rx_cytoplasm-( y_cytoplasm-rx_cytoplasm ) ).*rand( 1,1 );

        [ ~,ind1 ] = min( abs( xx-( x_ribosome+r_ribosome ) ) );
        [ ~,ind2 ] = min( abs( xx-( x_ribosome-r_ribosome ) ) );
        [ ~,ind3 ] = min( abs( yy-( y_ribosome+r_ribosome ) ) );
        [ ~,ind4 ] = min( abs( yy-( y_ribosome-r_ribosome ) ) );

        z1 = z_cytoplasm( ind3,ind1 );
        z2 = z_cytoplasm( ind3,ind2 );
        z3 = z_cytoplasm( ind4,ind1 );
        z4 = z_cytoplasm( ind4,ind2 );

        if ~( ( z1>0 )&&( z2>0 )&&( z3>0 )&&( z4>0 ) )
            continue
        end

        z_ribosome = r_ribosome^2-( YY-y_ribosome ).^2-( XX-x_ribosome ).^2;
        z_ribosome( z_ribosome<0 ) = 0;

        ribosome_phase1( yind,xind ) = ribosome_phase1( yind,xind ) + 2*pi/lam1*( n_ribosome-n_cytoplasm )*2*real( sqrt( z_ribosome ) );
        ribosome_phase2( yind,xind ) = ribosome_phase2( yind,xind ) + 2*pi/lam2*( n_ribosome-n_cytoplasm )*2*real( sqrt( z_ribosome ) );

        [ ~,x0 ] = min( abs( x( xind )-x_ribosome ) );
        [ ~,y0 ] = min( abs( y( yind )-y_ribosome ) );
        zmax = sqrt( z_cytoplasm( y0,x0 ) );
        z0 = -zmax + 2*zmax*rand( 1 );
        temp1 = real( sqrt( z_ribosome ) );
        for ii = 1:size( temp1,1 )
            for jj = 1:size( temp1,2 )
                [ ~,z1 ] = min( abs( z-z0-temp1( ii,jj ) ) );
                [ ~,z2 ] = min( abs( z-z0+temp1( ii,jj ) ) );
                if temp1( ii,jj ) > 0
                    nnc1( ii,jj,z2:z1 ) = n_ribosome;
                    nnc2( ii,jj,z2:z1 ) = n_ribosome;
                end
            end
        end
    end

    % optical path length through cytoplasm and nucleus
    opd_cytoplasm1( yind,xind ) = cytoplasm_phase1( yind,xind )*lam1/( 2*pi );
    opd_nucleus1( yind,xind ) = nucleus_phase1( yind,xind )*lam1/( 2*pi );
    opd_mitochondria1( yind,xind ) = mitochondria_phase1( yind,xind )*lam1/( 2*pi );
    opd_ribosome1( yind,xind ) = ribosome_phase1( yind,xind )*lam1/( 2*pi );

    opd_cytoplasm2( yind,xind ) = cytoplasm_phase2( yind,xind )*lam2/( 2*pi );
    opd_nucleus2( yind,xind ) = nucleus_phase2( yind,xind )*lam2/( 2*pi );
    opd_mitochondria2( yind,xind ) = mitochondria_phase2( yind,xind )*lam2/( 2*pi );
    opd_ribosome2( yind,xind ) = ribosome_phase2( yind,xind )*lam2/( 2*pi );

    % absorption through cytoplasm and nucleus
    absorption_cytoplasm1( yind,xind ) = W_cytoplasm*alpha_water*opd_cytoplasm1( yind,xind );
    absorption_nucleus1( yind,xind ) = W_nucleus*alpha_water*opd_nucleus1( yind,xind );
    absorption_mitochondria1( yind,xind ) = W_mitochondria*alpha_water*opd_mitochondria1( yind,xind );
    absorption_ribosome1( yind,xind ) = W_ribosome*alpha_water*opd_ribosome1( yind,xind );

    absorption_cytoplasm2( yind,xind ) = W_cytoplasm*alpha_water*opd_cytoplasm2( yind,xind );
    absorption_nucleus2( yind,xind ) = W_nucleus*alpha_water*opd_nucleus2( yind,xind );
    absorption_mitochondria2( yind,xind ) = W_mitochondria*alpha_water*opd_mitochondria2( yind,xind );
    absorption_ribosome2( yind,xind ) = W_ribosome*alpha_water*opd_ribosome2( yind,xind );

    % total absorption
    absorption1( yind,xind ) = absorption_nucleus1( yind,xind )+absorption_cytoplasm1( yind,xind )+absorption_mitochondria1( yind,xind )+absorption_ribosome1( yind,xind );
    absorption2( yind,xind ) = absorption_nucleus2( yind,xind )+absorption_cytoplasm2( yind,xind )+absorption_mitochondria2( yind,xind )+absorption_ribosome2( yind,xind );

    cell_flag = 1;

    T1 = exp( -1*absorption1 );% transmittance according to Beer's law
    T2 = exp( -1*absorption2 );% transmittance according to Beer's law
    image_intensity1 = T1*I0; % scale intensity
    image_intensity2 = T2*I0; % scale intensity
    image_phase1 = ( cytoplasm_phase1 + nucleus_phase1 + ribosome_phase1 + mitochondria_phase1 ); % aggregate phase
    image_phase2 = ( cytoplasm_phase2 + nucleus_phase2 + ribosome_phase2 + mitochondria_phase2 ); % aggregate phase
    opd1 = image_phase1*lam1/( 2*pi ); % aggregate optical path length difference
    opd2 = image_phase2*lam2/( 2*pi ); % aggregate optical path length difference

    NA = 0.25;
    ds_rate = floor( lam1/( 2*NA )/d );

    x = ( 0:size(  image_phase1,2 )-1 )*d;
    y = ( 0:size(  image_phase1,1 )-1 )*d;
    [ X,Y ] = meshgrid( x,y ); % spatial grid
    opd1 = image_phase1*lam1/( 2*pi ); % aggregate optical path length difference
    opd2 = image_phase2*lam2/( 2*pi ); % aggregate optical path length difference


    nc1( xind,yind,: ) = nnc1;
    nc2( xind,yind,: ) = nnc2;

    int_nc1 = zeros( [ size( nc1,1 ) size( nc1,2 ) ] );
    int_nc2 = zeros( [ size( nc2,1 ) size( nc1,2 ) ] );
    dz = z( 2 )-z( 1 );
    for ii = 1:size( nc1,1 )
        disp( [ num2str( ii ) ' of ' num2str( size( nc1,1 ) ) ] )
        for jj = 1:size( nc1,2 )
            int_nc1( ii,jj ) = 1/( 2*hz( ii,jj ) )*trapz( z,squeeze( nc1( ii,jj,: ) ) );
            int_nc2( ii,jj ) = 1/( 2*hz( ii,jj ) )*trapz( z,squeeze( nc2( ii,jj,: ) ) );
        end
    end

    % complex phase object
    im1 = sqrt( image_intensity1 ).*exp( 1i*image_phase1 );
    im2 = sqrt( image_intensity2 ).*exp( 1i*image_phase2 );


    K = 5;
    image_phase1 = 2*medfilt2( image_phase1,[ K K ] );
    image_phase2 = 2*medfilt2( image_phase2,[ K K ] );
    image_intensity1 = medfilt2( image_intensity1,[ K K ] );
    image_intensity2 = medfilt2( image_intensity2,[ K K ] );
    int_nc1 = medfilt2( int_nc1,[ K K ] );
    int_nc2 = medfilt2( int_nc2,[ K K ] );
    hz = 2*medfilt2( 2*hz,[ K K ] );

    maxPhase = 20;
    maxIndex = 3;
    nc1_conv = imgaussfilt3( nc1,2 );
    nc2_conv = imgaussfilt3( nc2,2 );


    nc1_conv = nc1_conv - n_medium1;
    for iz = 1:length( z )
        t = Tiff( [ pwd filesep 'cell_phantoms_' num2str(M) 'x' num2str(N) filesep 'phantomIndex3D' filesep 'refractiveIndex_r' num2str( iRealization,'%03d' ) '_z' num2str( iz,'%03d' ) '.tif' ],'w' );
        tagstruct.ImageLength     = size(nc1,1);
        tagstruct.ImageWidth      = size( nc1,2);
        tagstruct.Photometric     = Tiff.Photometric.MinIsBlack;
        tagstruct.BitsPerSample   = 16;
        tagstruct.SamplesPerPixel = 1;
        tagstruct.RowsPerStrip    = 16;
        tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
        tagstruct.Software        = 'MATLAB';
        t.setTag(tagstruct)
        t.write( uint16( 1e5*squeeze( nc1_conv( :,:,iz ) ) ) );
        t.close();
    end

    t = Tiff( [ pwd filesep 'cell_phantoms_' num2str(M) 'x' num2str(N) filesep 'phantomPhase_505nm' filesep 'phantom_phase_505nm_' num2str( iRealization ) '.tif' ],'w' );
    tagstruct.ImageLength     = size(image_phase1,1);
    tagstruct.ImageWidth      = size(image_phase1,2);
    tagstruct.Photometric     = Tiff.Photometric.MinIsBlack;
    tagstruct.BitsPerSample   = 32;
    tagstruct.SamplesPerPixel = 1;
    tagstruct.RowsPerStrip    = 16;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagstruct.Software        = 'MATLAB';
    t.setTag(tagstruct)
    t.write( uint32( 1e6*image_phase1 ) );
    t.close();

    t = Tiff( [ pwd filesep 'cell_phantoms_' num2str(M) 'x' num2str(N) filesep 'phantomPhase_488nm' filesep 'phantom_phase_488nm_' num2str( iRealization ) '.tif' ],'w' );
    tagstruct.ImageLength     = size(image_phase2,1);
    tagstruct.ImageWidth      = size(image_phase2,2);
    tagstruct.Photometric     = Tiff.Photometric.MinIsBlack;
    tagstruct.BitsPerSample   = 32;
    tagstruct.SamplesPerPixel = 1;
    tagstruct.RowsPerStrip    = 16;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagstruct.Software        = 'MATLAB';
    t.setTag(tagstruct)
    t.write( uint32( 1e6*image_phase2 ) );
    t.close();

    t = Tiff( [ pwd filesep 'cell_phantoms_' num2str(M) 'x' num2str(N) filesep 'phaseHeight' filesep 'phantom_height_' num2str( iRealization ) '.tif' ],'w' );
    tagstruct.ImageLength     = size(hz,1);
    tagstruct.ImageWidth      = size(hz,2);
    tagstruct.Photometric     = Tiff.Photometric.MinIsBlack;
    tagstruct.BitsPerSample   = 32;
    tagstruct.SamplesPerPixel = 1;
    tagstruct.RowsPerStrip    = 16;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagstruct.Software        = 'MATLAB';
    t.setTag(tagstruct)
    t.write( uint32( 1e9*hz ) );
    t.close();

    t = Tiff( [ pwd filesep 'cell_phantoms_' num2str(M) 'x' num2str(N) filesep 'phaseIndex' filesep 'phantom_index_' num2str( iRealization ) '.tif' ],'w' );
    tagstruct.ImageLength     = size(int_nc1,1);
    tagstruct.ImageWidth      = size( int_nc1,2);
    tagstruct.Photometric     = Tiff.Photometric.MinIsBlack;
    tagstruct.BitsPerSample   = 32;
    tagstruct.SamplesPerPixel = 1;
    tagstruct.RowsPerStrip    = 16;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagstruct.Software        = 'MATLAB';
    t.setTag(tagstruct)
    t.write( uint32( 1e6*int_nc1 ) );
    t.close();

    t = Tiff( [ pwd filesep 'decoupled_images_unwrap' filesep 'phantom_index_488nm_' num2str( iRealization ) '.tif' ],'w' );
    tagstruct.ImageLength     = size(int_nc2,1);
    tagstruct.ImageWidth      = size( int_nc2,2);
    tagstruct.Photometric     = Tiff.Photometric.MinIsBlack;
    tagstruct.BitsPerSample   = 32;
    tagstruct.SamplesPerPixel = 1;
    tagstruct.RowsPerStrip    = 16;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagstruct.Software        = 'MATLAB';
    t.setTag(tagstruct)
    t.write( uint32( 1e6*int_nc2 ) );
    t.close();

end